> 规范源：EPlayground/port2080/VLLM-DFLASH2-HANDOFF.md（本文为随分支副本）

# VLLM-DFLASH2-HANDOFF.md — 给下一个 agent：在 vLLM 上适配 DFlash2

> 写于 2026-08-25。作者：sglang Turing 移植 + DFlash2 打通全程的接手 agent。
> 目标读者：即将在 `EPlayground/vLLM-2080Ti-Definitive`（vllm-2080ti-definitive-0.2.x，SM75 深改 fork）上实现 DFlash2 投机解码的 agent。
> 本文是我们踩坑 35+ commits 换来的全部经验教训，按"你会遇到的顺序"组织。**先读完本文再动代码。**

---

## 0. 任务是什么

DFlash2 = 扩散式草稿投机解码：目标模型在中继层导出 hidden states → 草稿模型（W8 量化、bf16 原生、block_size=8）用 selector 从 `<|MASK|>` 起逐位扩散出候选块 → 目标模型一次 verify 整块、贪心链式接受。线性链 topk=1，无树。

- **已验证性能（sglang，dual 2080 Ti 22GB×2 NVLink，TP2）**：单流 DECODE **71.6–72.4 tok/s**（AR 基线 29.86 的 2.4×，MTP3 NEXTN×3 36.35 的 ~2×）；bs=2 聚合 **123.7 tok/s**；accept len 7.42–7.97。
- **你的对标物**：`audit-vllm-fork.md` 记录该 vLLM fork 已有 MTP 投机解码的 CUDA-graph 路径 + GDN forward CUDA kernel（目标模型的混合线性注意力已在跑）+ TurboQuant KV + FlashInfer SM75 路由。你要做的是把草稿一侧换成 DFlash2。
- **参考实现三处**：
  1. sglang 完整生产实现：分支 `TnzGit/sglang:turing-sm75`（commit 4227b13），文件索引见 §2；
  2. 上游参考代码：`port2080/reference-dflash2/`（qwen3_dflash2.py / speculator.py / up_*）；
  3. 静态对照报告：`port2080/dflash2-static-diff.md`。

## 1. 模型与硬件事实（先记住再写代码）

| 项 | 值 |
|---|---|
| 目标模型 | lued-Qwen3.8-INT8-DFlash2（Qwen3_5ForConditionalGeneration，compressed-tensors W8A16 INT8，hybrid GDN+full-attn） |
| 草稿模型 | lued-DFlash2-W8-draft（DFlash2DraftModel，W8 CT 打包权重，**bf16 原生检查点**，block_size=8） |
| mask token | `<|MASK|>` id=248070（运行时确认过 override） |
| 硬件 | 2× RTX 2080 Ti 22G 改装版 + NVLink；sm_75；无 bf16 tensor core 但 CUDA core/Triton bf16 可用 |
| 显存预算 | 总 22.0 GiB；目标权重 14.68 + 草稿 1.44 = 16.12 GB 固定占用；profile 时空闲恒 4.98 GB |

## 2. sglang 参考实现的文件索引（照着抄逻辑）

```
python/sglang/srt/
├── speculative/
│   ├── dflash_worker_v2.py        # 核心 worker：draft propose/verify 编排、folded sampler、
│   │                              #   _SelectorDraftSampler（图内采样）、静态缓冲区语义
│   ├── dflash_utils.py            # 工具全集：dflash_draft_cell_size_per_token（KV 定价）、
│   │                              #   compute_dflash_correct_drafts_and_bonus（triton 接受核，可直接复用）、
│   │                              #   build_target_layer_ids（中继层选取）、table_qk_norm_rope_（fused QKV rope）
│   ├── dflash_info_v2.py          # verify 元信息
│   └── draft_worker_common.py     # draft TP worker 构建
├── models/dflash.py               # DFlash2DraftModel 定义
├── mem_cache/kv_cache_configurator.py   # 内存预算求解（含我们加的 Mamba budget solve 日志）
├── model_executor/pool_configurator.py  # cell_size 定价（DFLASH draft cell 加算）
└── kernels/ops/speculative/dflash.py    # 自有 triton 核
```

关键 commit（教训都挂在对应 fix 上）：`e69576e` 权重加载、`ec2009b` 精度、`ab86792`+`4227b13` 探针门控、`79c136f` 预算日志。

## 3. 六大教训（按你会遇到的顺序）

### L1 · 量化打包权重的静默丢失（最先撞）

W8 草稿的 fc 权重是 compressed-tensors **打包格式**。用裸 `nn.Linear` 加载会静默丢掉它们→随机权重初始化→草稿输出是噪声但流程不报错。sglang 修复（e69576e）：CT 感知 Linear + **加载器按量化名硬校验**（缺 quant 名直接 fail fast）。
**vLLM 动作**：确认草稿模型注册到 compressed-tensors 反序列化路径；加一个"权重全零/随机检测"的自检（对比同名张量 hash 与 checkpoint）。

### L2 · bf16 原生检查点在 fp16 下必然溢出（最隐蔽，耗了我们两天）

bf16 原生激活天然达 |x|~1e5 > fp16 max 65504 → 每步溢出 → NaN → nan_to_num 清零带伤运行 → **accept 卡 1.15、输出看似正常**。症状极具迷惑性：流程通、文本连贯、就是不接受。
**修复原则（ec2009b）**：
- 草稿整体跑 **bfloat16**（sm75 无 bf16 TC 也能走 CUDA-core/Triton 数值路径，慢一点但正确）；
- **选择性 cast**：quant 打包参数（weight_packed/weight_scale/weight_shape/global_scale）必须保持量化 kernel 构建时的 dtype（fp16），只 cast 其余；
- rope/cos_sin_cache 一律 fp32 域；
- 验收手段：p8 harness 式张量级对照（纯 torch 重算 vs 引擎 dump，final hidden rel ~4e-4）。
**vLLM 动作**：草稿 runner 的 dtype 解析默认 bfloat16；禁止全局 `.half()`；对 marlin/fp8 packed 参数设 cast 白名单。

### L3 · 调试探针纪律（图捕获杀手 + 11% 隐形税）

- 任何捕获区内的 `float(...)`/`.item()` 强制 D2H 同步 → `cudaErrorStreamCaptureInvalidated`，图捕获直接炸；
- 未门控的每步 WARNING 日志：单基准刷 1250–14676 行，实测拖慢 **~11%**（p19 归因实验：62.82→69.89 纯靠移除探针税）。
**规则**：所有 DFLASHDBG 类输出必须 `env 门控 + 次数上限（计数器<2/3/4）` 双保险；新代码 review 时 grep `_l.getLogger` 和 `.item()`。

### L4 · 图捕获与 sampler 折叠

- 顺序：目标权重 → 草稿权重 → 内存池 → attention 后端 → 目标 verify 图 → 草稿图（含 folded selector）；
- folded sampler（`_SelectorDraftSampler`）：static 缓冲（out/temperatures/greedy_mask/uniforms/candidate_out/q_out）地址烘进图，host 在 replay 前 stage 采样参数，greedy 用 mask 选择 argmax——一套图服务贪心+采样；
- **数值零漂移验证法**：同参数 graphs vs eager 各跑固定 prompt 贪心，比对输出 md5。我们实测完全一致（e498ed2c…）。你也应该把这个做成回归项。

### L5 · 小显存 + hybrid mamba 的内存预算（322 事故全文）

现象：KV 池被算到只剩 322 token。真相与投机算法无关，是预算挤占：

```
KV预算 = profile时空闲 − pre_model_load×(1−memfrac) − mamba扣减
mamba扣减 = (K+1)·per_req[主态] + (min(N,K//ratio)+1)·D·per_req[verify中间态]
per_req(fp16 ssm) ≈ 37.4 MB/rank，(fp32) ≈ 66 MB；D=draft_tokens=8；ratio=5(base3+overlap extra_buffer2)
```

- verify 中间态 scratch 是**真实物理分配**（每请求每 draft 位一份完整时间态），D=8/N=4 时高达 1.5 GB——别幻想它是幻影双计；
- **并发钳制**：`max_running_requests ≤ K//ratio`；
- **radix 保持规则**：K ≥ ratio×(N+1)，否则最老前缀丢 mamba 状态→下轮整段重预填（11.6s→4.5s 的差距）；
- 诊断手段：我们在 kv_cache_configurator 加了 `Mamba budget solve:` 分解日志（79c136f），vLLM 侧建议同样做。
**vLLM 动作**：为草稿 KV 单独定价时用 `dflash_draft_cell_size_per_token` 同款公式（kv_heads×(head_dim+v_head_dim)×layers×dtype_size，TP 分片后）；目标池预算要加上这份加算。

### L6 · 基准方法论（否则你会在假数字上浪费一周）

1. **radix/前缀缓存污染计时**：同一 prompt 跑两遍取时间差的做法会得到负 decode（前缀命中让第二遍几乎只剩 decode）。用流式 ttft/e2e 口径（我们的 p6_bench.sh：`ct/(e2e−ttft)`，distinct prompts）。
2. **accept len 是主题强相关量**：四个标准主题中 space exploration 单流只有 **6.05**，其余 7.4–7.97。混批日志均值会被拉低（bs=4 看到 6.8 属正常）。跨配置对比必须锁死主题集。
3. **历史基线可能带税**：对比前先同代码 A/B（我们的 62.82 旧基线含 ~11% 探针税，真实 eager 是 ~70）。
4. **冷启动预填饿死解码**：无 mixed-chunk 时，1024-token 预填块（~1.65s/块）期间解码流完全停摆（末位流可跌至 5 tok/s）。测稳态吞吐要么暖缓存要么剔除 TTFT 段。
5. 正确姿势样板：`port2080/p18_compare.sh`（多臂同代码对比 + 贪心 md5 校验）。

## 4. 当前成绩单（你要对标/超越的数字）

| 场景 | sglang DFlash2 | 对照 |
|---|---:|---|
| 单流 DECODE | 71.6–72.4 tok/s | AR 29.86；MTP3 36.35 |
| bs=2/3/4 聚合 | 123.7 / 123.4 / 114.5 tok/s | 带宽折算参考 ~71（单流的 101%）|
| accept len | 7.42–7.97（space exploration 除外 ~6.05）| |
| PREFILL / TTFT | ~624 tok/s / 7.3s @4K | |
| KV 池 | 58991 token（fp16 ssm/0.92/K=10）| |

## 5. vLLM 侧切入点建议（基于 audit-vllm-fork.md）

1. **投机框架挂点**：fork 已有 MTP 的 spec-decode CUDA-graph 路径（§审计 2.x/相关节）。DFlash2 的差异点：草稿输入不是 target logits 而是**中继层 hidden states**（需 target 模型 hook 导出多层 hidden——sglang 用 `build_target_layer_ids` 选层 + CaptureHiddenMode.FULL）；
2. **接受规则**：线性链贪心 `candidates[:,1:]==target_predict[:,:-1]` 连续匹配 + bonus——直接移植 `compute_dflash_correct_drafts_and_bonus` triton 核（自包含，仅依赖 candidates/target_predict 两张量）；
3. **草稿 KV**：独立小池（cell ~10 KB/token），或评估复用 vLLM KV 分配器页机制；
4. **kernel 复用清单**（都在 dflash_utils.py，无 sglang 重依赖）：`_fused_correct_drafts_and_bonus_kernel`、`_table_qk_norm_rope_kernel`、selector lattice 逻辑在 models/dflash.py；
5. **Triton JIT 预热**：高 bs 首请求会现编译（我们见到 42 条 lazy-load 警告、空闲内存一度 0.47 GiB），engine init 后按捕获 bs 列表空跑一轮。

## 6. 验收清单（照抄，一项不过就别报完成）

- [ ] 启动日志：profiled KV ≥ 50k（memfrac 0.92 档）；`Mamba budget solve` 分解行合理
- [ ] 单流 DECODE ≥ 63.6 tok/s（p6 流式口径，3×4096 distinct/128，贪心）
- [ ] accept len 7.4–7.97（除 space exploration 外的主题）
- [ ] 贪心输出与 eager 路径 md5 一致（fold 回归）
- [ ] 10 连发长请求零 400、零 NaN/LAYER-OUT
- [ ] 并发 2 聚合 ≥ 110 tok/s（可选：bs4 ≥ 105，需 K≥25 规则）
- [ ] 张量级对照 rel ≤ 1e-3（harness 法）

## 7. 运维红线与速查坑

- 主机 `ssh dual2080ti@dual2080ti`；venv `source ~/.venv-vllm-dflash2-upstream/bin/activate`（忘了 source 会报 `nohup: failed to run command 'python'`——先看日志首行再怀疑环境）；
- **8000 端口=生产禁触**，测试一律 8002；起服务前 `nvidia-smi` 确认空闲；
- 内联 ssh 命令里不要出现 `pkill -f "sglang.launch_server"` 同款字符串（模式匹配到 ssh 自身命令行→自杀 exit 255），放进脚本文件执行；
- 模型路径：`~/models/lued-Qwen3.8-INT8-DFlash2`、`~/models/lued-DFlash2-W8-draft`；
- `SGLANG_SIMULATE_ACC_LEN`（及未来 vLLM 等价物）若被设置会伪造接受率——排查前先查 env；
- sglang 侧改动同步主机用 scp 单文件覆盖 `~/sglang`（工作区即部署态，勿在主机 git 操作）。
