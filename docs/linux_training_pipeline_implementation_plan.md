# Linux 迁移与大模型训练侧完整施工方案

> 文档状态：施工依据（待实施）  
> 编制日期：2026-07-15  
> 适用工程：`Jupedsim` / `jupedsim_mall`  
> 目标：将现有工程迁移到 Linux GPU 服务器，并补齐“真实数据 -> 训练样本 -> SFT/DPO -> 模型服务 -> 仿真实验 -> 论文报告”的完整、可复现流水线。

## 1. 当前事实与目标边界

### 1.1 当前工程已经具备的能力

- JuPedSim 仿真、场景配置、规则基线和 LLM 路径规划接口。
- OpenAI-compatible `/v1/chat/completions` 调用能力。
- ATC、LLMob 等真实数据的读取、切分、画像/参考统计和回放相关处理。
- 实验矩阵、运行清单、指标汇总、绘图、报告和质量门禁。
- 当前已有报告主要用于验证工程流水线可运行，不代表模型已由真实数据训练，也不能直接作为最终论文结论。

### 1.2 当前工程缺失的训练闭环

- 没有将真实轨迹稳定转换为 SFT/DPO 样本的数据契约。
- 没有教师模型候选生成、候选评分和偏好对构建。
- 没有本地基础模型、LoRA/QLoRA、SFT、DPO 训练入口。
- 没有训练集泄漏检查、训练指标、模型登记和适配器版本追踪。
- 没有将训练后的模型部署为现有仿真可调用 API 的正式步骤。
- 当前场景配置中存在固定本地模型地址和模型名，不适合服务器多模型实验。

### 1.3 本次建设后的目标架构

```text
ATC / LLMob 原始数据
        |
        v
清洗、坐标标定、切分、画像、轨迹事实提取
        |
        v
训练样本构建 -----> 教师 API 生成候选（只用于标签/偏好数据）
        |                           |
        +---------- 候选校验与评分 -+
        |
        +--> SFT JSONL --> QLoRA-SFT adapter
        |
        +--> DPO JSONL --> QLoRA-DPO adapter
                               |
                               v
                   vLLM/SGLang OpenAI-compatible API
                               |
                               v
                    现有 LLMRoutePlanner / JuPedSim
                               |
                               v
                    成对实验、统计分析、论文报告
```

训练后的大模型仍然可以通过 API 接入。区别在于：API 后端不再只能是第三方通用模型，而是 Linux 服务器上加载“基础模型 + 本项目适配器”的本地服务。

### 1.4 与外部工程方法的关系

外部参考工程采用了可复用的混合路线：用 DashScope/OpenAI-compatible 教师 API 生成候选和偏好数据，在本地 Qwen 模型上先做 LoRA/QLoRA SFT、再做 DPO，最后通过 Transformers/PEFT 加载 adapter 推理。本方案保留“教师 API 造数据 + 本地轻量微调 + SFT 后 DPO”这条主线，但不直接复制其脚本，原因包括：

- 外部脚本存在 Linux 绝对路径和人工 ID 列表，不具备跨机器数据契约。
- 训练和推理的默认基础模型规格存在不一致风险，容易加载错误 adapter。
- SFT/DPO 的独立评估、数据泄漏审计和训练 manifest 不够完整。
- 外部推理直接嵌入模型，本工程已有稳定 API 边界，更适合把本地训练模型部署成 OpenAI-compatible 服务。
- 本工程还需要把训练结果放回 JuPedSim 做系统层配对实验，不能只比较训练 loss 或离线路线分数。

因此，本方案是对外部方法的工程化重建：保留有效训练范式，补上真实数据治理、可追溯性、服务化和论文实验门禁。

## 2. 总体实施原则

1. **先冻结数据契约，再训练。** 坐标、切分、速度阈值或标签定义变化后，旧训练集必须作废并重新生成。
2. **先做 SFT，再做 DPO。** SFT 负责输出格式和基础行为，DPO 负责偏好对齐；第一版不直接上强化学习。
3. **按人员切分后再派生样本。** 同一人员的窗口、画像、候选和统计量不能跨 train/tuning/evaluation。
4. **训练选择只看 tuning。** evaluation 只在模型、参数和统计方案冻结后执行一次正式评估。
5. **仿真与训练解耦。** 仿真进程只调用 API；模型权重、CUDA 和训练依赖不进入仿真环境。
6. **每个产物必须可追溯。** 数据、提示词、评分器、基础模型、适配器、代码提交和运行环境均记录哈希或版本。
7. **不训练隐藏思维过程。** 数据中保存结构化决策和必要的短理由，不要求教师模型输出长篇推理过程。

## 3. Linux 服务器目录规划

建议不要把大数据、模型权重和实验输出全部放在 Git 工作树内。

```text
/srv/jupedsim-mall/                 # Git 工程代码
/srv/jupedsim-data/
  raw/                              # 只读原始数据
  interim/                          # 可重建中间数据
  processed/                        # 冻结后的数据集
/srv/jupedsim-artifacts/
  datasets/                         # SFT/DPO JSONL 与 manifest
  checkpoints/                      # 训练断点
  adapters/                         # 发布的 LoRA adapter
  model-registry/                   # 模型登记信息
  experiments/                      # 仿真运行与报告
/srv/models/                        # 基础模型缓存或离线快照
/srv/jupedsim-cache/
  huggingface/
  torch/
```

推荐环境变量：

```bash
export JUPEDSIM_MALL_PROJECT_ROOT=/srv/jupedsim-mall
export JUPEDSIM_MALL_DATA_ROOT=/srv/jupedsim-data
export JUPEDSIM_MALL_OUTPUT_ROOT=/srv/jupedsim-artifacts/experiments
export JUPEDSIM_MALL_ARTIFACT_ROOT=/srv/jupedsim-artifacts
export JUPEDSIM_MALL_MODEL_ROOT=/srv/models
export HF_HOME=/srv/jupedsim-cache/huggingface
export TORCH_HOME=/srv/jupedsim-cache/torch
```

工程目前主要支持 `JUPEDSIM_MALL_PROJECT_ROOT`。其余根目录变量需要按第 7 节改造后才正式生效。

权限建议：

```bash
sudo groupadd -f jupedsim
sudo chgrp -R jupedsim /srv/jupedsim-*
sudo chmod -R 2770 /srv/jupedsim-data /srv/jupedsim-artifacts
sudo chmod -R 2755 /srv/jupedsim-mall
```

ATC/LLMob 原始数据不得提交到 Git。若数据许可限制共享，应将 `/srv/jupedsim-data/raw` 设为课题组授权用户可读。

## 4. 服务器环境与双虚拟环境

### 4.1 基础环境清单

- Ubuntu 22.04 LTS 或 24.04 LTS，x86_64。
- NVIDIA 驱动、可见 GPU 和足够显存。
- Python 3.11/3.12，具体版本以 JuPedSim、PyTorch、vLLM 的共同支持范围为准。
- Git、Git LFS（如模型登记需要）、rsync、curl、build-essential、sqlite3。
- 磁盘必须同时容纳基础模型、量化缓存、训练断点和实验输出。

上线前记录：

```bash
uname -a
cat /etc/os-release
nvidia-smi
df -h
free -h
python3 --version
git --version
```

不要在文档中固定 CUDA/PyTorch 安装命令。部署时应根据服务器驱动，使用 PyTorch 官方安装选择器确认兼容版本，并将最终版本写入锁文件和环境清单。

### 4.2 仿真环境

```bash
cd /srv/jupedsim-mall
python3 -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
python -m pytest -q
```

### 4.3 训练环境

训练侧单独使用 `.venv-train`，防止 JuPedSim、PyTorch、bitsandbytes、Transformers 和 vLLM 依赖互相限制。

计划新增 `training` 可选依赖组，至少包含：

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `trl`
- `bitsandbytes`（NVIDIA QLoRA）
- `safetensors`
- `tensorboard` 或同类本地日志工具

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
# 先安装与驱动匹配的 PyTorch，再安装工程训练依赖。
python -m pip install -e ".[training]"
```

如 vLLM 与训练依赖冲突，再拆出 `.venv-serve`。生产实验中，训练和服务不能同时争用同一张 GPU，除非已经验证显存预算。

## 5. 从 Windows 迁移到 Linux

### 5.1 代码迁移

首选通过 Git 迁移代码：

```bash
git clone <repository-url> /srv/jupedsim-mall
cd /srv/jupedsim-mall
git checkout <frozen-commit-or-branch>
git status --short
```

若当前修改尚未进入远程仓库，可先在 Windows 创建提交或补丁，再传输。不要复制以下内容：

- `.venv*`
- `__pycache__`
- `.pytest_cache`
- Hugging Face 缓存
- 可重建的旧实验输出
- Windows 专用二进制依赖

### 5.2 数据迁移与校验

示例：

```bash
rsync -avh --partial --append-verify \
  /source/atc-20121114.csv \
  user@server:/srv/jupedsim-data/raw/atc-20121114/

ssh user@server \
  'sha256sum /srv/jupedsim-data/raw/atc-20121114/atc-20121114.csv'
```

当前已知 ATC 文件的预期 SHA-256 应在实际迁移前从现有 provenance 再核对一次；不能仅凭文件名判定数据相同。数据 manifest 应记录：

- 原始文件名、字节数、SHA-256。
- 数据许可和来源说明。
- 导入日期和导入人。
- 坐标单位、速度单位和时间单位。
- 区域文件及其 SHA-256。

### 5.3 Linux 特有检查

- Git 中脚本统一使用 LF 行尾。
- Bash 脚本设置可执行位。
- 所有导入路径检查大小写；Linux 文件名大小写敏感。
- 不在配置中写 `D:\...` 或 `E:\...`。
- 图形输出使用 Matplotlib `Agg` 后端，服务器不依赖桌面环境。
- 所有运行命令从任意工作目录执行时都能正确解析项目根目录。

## 6. 训练前必须修复的数据问题

以下项目属于硬门禁，未通过时不得生成正式训练集。

### 6.1 坐标系标定

当前真实轨迹坐标直接用于项目区域判断，隐含“ATC 坐标和仿真地图坐标已经对齐”的假设。应新增显式坐标变换配置：

```yaml
coordinate_transform:
  source_crs: atc_local_mm
  target_crs: mall_local_m
  scale: [0.001, 0.001]
  rotation_deg: 0.0
  translation_m: [0.0, 0.0]
  axis_order: xy
  flip_x: false
  flip_y: false
  version: atc_to_mall_v1
```

验收标准：

- 抽样轨迹叠加在区域图上，入口、商铺、通道和出口位置合理。
- 区域覆盖率、越界率、跳跃率和区域转移矩阵有自动报告。
- 变换配置及哈希写入数据集 manifest。

### 6.2 速度阈值统一

当前 ATC 参考处理中低速阈值与仿真指标中的阈值不一致。应建立单一指标定义文件，例如：

```yaml
metrics_version: mobility_metrics_v2
slow_speed_threshold_mps: 0.20
stationary_speed_threshold_mps: 0.05
min_track_duration_s: 10.0
```

训练标签、真实参考、仿真指标和报告必须引用同一版本。阈值变更后，所有派生产物失效。

### 6.3 人员抽样语义

当前部分流程虽记录 `sampling_seed`，但 `max_persons` 可能表现为“取前 N 个合格人员”，不是随机抽样。应选择并固化一种语义：

- `selection=sorted_first`：删除误导性的随机种子。
- `selection=seeded_random`：按人员 ID 形成稳定候选集后，用固定种子抽样。

正式实验推荐 `seeded_random`，并保存入选人员 ID 的哈希列表。

### 6.4 数据切分和泄漏审计

保留按人员稳定分桶的思想，并明确：

```text
evaluation: bucket 0
tuning:     bucket 1
train:      bucket 2..9
```

实际比例和哈希算法写入配置。必须在轨迹窗口、画像、参考统计和教师候选生成之前完成切分。新增自动检查：

- 三个分区人员 ID 交集为空。
- 同一原始轨迹不能产生跨分区窗口。
- evaluation 的画像统计不能参与提示词模板、评分器权重或超参数选择。
- 训练样本中不能出现 evaluation 的人员 ID、轨迹摘要或目标序列。

### 6.5 历史输出隔离

`outputs/pipeline_acceptance` 继续保留为工程验收样例，但应标记：

```text
result_class = engineering_smoke
trained_model = false
paper_eligible = false
```

正式训练后实验必须输出到新的 run root，不能覆盖旧报告。

## 7. 现有工程需要修改的文件和行为

| 范围 | 修改内容 | 验收结果 |
|---|---|---|
| `src/jupedsim_mall/project.py` | 增加 data、artifact、output、model 根目录解析 | Linux 绝对路径可由环境变量注入 |
| `pyproject.toml` | 增加 `training` 可选依赖；避免将 CUDA 专用 wheel 写死到通用依赖 | 仿真环境和训练环境可独立安装 |
| `requirements-lock.txt` | 保留 Windows 基线；新增 Linux 仿真/训练锁文件或统一 lock 工具 | 能重建服务器环境 |
| `configs/scenarios/*llm*.json` | 移除固定 `127.0.0.1:8600` 和固定模型名，改为模型注册表或运行时注入 | 同一场景可切换 BASE/SFT/DPO |
| `scripts/run_experiment_suite.py` | 明确环境变量优先级；记录脱敏后的实际生效配置 | 服务器变量不会被场景静默覆盖 |
| `src/jupedsim_mall/llm_prompts.py` | 提示词模板版本化，并允许训练/部署共同读取 | 训练提示和在线提示一致 |
| `schemas/llm_route_response.schema.json` | 保持为输出真值契约；变更时升级 schema 版本 | 数据构建、推理和评估共用校验器 |
| `src/jupedsim_mall/doctor.py` | 增加 Linux、GPU、磁盘、模型服务、数据哈希检查 | 一条命令输出可训练性诊断 |
| `scripts/*.ps1` | 增加功能等价的 `.sh`；Python 入口继续作为真实实现 | Linux 无需 PowerShell |
| `.gitignore` | 增加模型、adapter、checkpoint、教师缓存和私密数据规则 | 大文件与密钥不会误提交 |
| `README.md` | 增加 Linux 快速开始、训练、服务和实验入口 | 新服务器可按文档复现 |

环境优先级建议固定为：

```text
CLI 显式参数 > 实验矩阵 model_ref > 进程环境变量 > 场景默认值
```

API key 永不写入 `resolved_config`。运行清单只记录 key 的来源名称和是否存在，例如 `LOCAL_LLM_API_KEY_present=true`。

## 8. 训练侧建议目录结构

```text
configs/
  data/
    atc_dataset_v1.yaml
  training/
    qwen_8b_qlora_sft_v1.yaml
    qwen_8b_qlora_dpo_v1.yaml
    plan_scorer_v1.yaml
  models/
    base_qwen_v1.yaml
    mall_sft_v1.yaml
    mall_sft_dpo_v1.yaml
schemas/
  training_example.schema.json
  dataset_manifest.schema.json
  training_run_manifest.schema.json
src/jupedsim_mall/training/
  __init__.py
  contracts.py
  dataset_builder.py
  candidate_generator.py
  candidate_validator.py
  plan_scorer.py
  leakage_audit.py
  train_sft.py
  train_dpo.py
  evaluate_model.py
  model_registry.py
scripts/
  build_training_dataset.py
  generate_teacher_candidates.py
  train_sft.py
  train_dpo.py
  evaluate_trained_model.py
  serve_model.sh
  run_quality_gate.sh
  run_experiment_suite.sh
tests/
  test_training_contracts.py
  test_dataset_builder.py
  test_plan_scorer.py
  test_leakage_audit.py
  test_model_registry.py
```

训练实现应复用现有路线响应 schema、区域可达性校验和画像定义，不能复制出另一套含义相近但不一致的规则。

## 9. 训练数据契约

### 9.1 通用样本字段

每个派生样本至少包含：

```json
{
  "example_id": "sha256:...",
  "dataset_version": "atc_plan_v1",
  "partition": "train",
  "source": {
    "dataset": "atc-20121114",
    "person_id_hash": "sha256:...",
    "window_start_s": 0.0,
    "window_end_s": 60.0
  },
  "contracts": {
    "prompt_version": "1.0",
    "response_schema_version": "1.0",
    "profile_version": "...",
    "region_map_hash": "sha256:...",
    "coordinate_transform_version": "atc_to_mall_v1",
    "metrics_version": "mobility_metrics_v2"
  },
  "input": {
    "profile": {},
    "available_regions": [],
    "available_exits": [],
    "observed_context": {}
  },
  "target": {
    "plans": []
  },
  "label": {
    "source": "ground_truth_projection|teacher|rule|human",
    "quality_flags": []
  }
}
```

原始人员 ID 默认只保存哈希；如必须回溯，映射表应存放在受限数据目录，不进入训练集和实验输出。

### 9.2 SFT 数据

最终 JSONL 使用与部署 tokenizer/chat template 兼容的 `messages` 结构：

```json
{
  "example_id": "...",
  "messages": [
    {"role": "system", "content": "<versioned system prompt>"},
    {"role": "user", "content": "<agent and map context>"},
    {"role": "assistant", "content": "<strict route-plan JSON>"}
  ]
}
```

第一阶段建议每个样本只包含一个 agent，并在仿真场景中设置对应的小批量调用，以降低输出缺人和错配风险。第二阶段再构建 batch size 为 1、4、8 的混合样本，覆盖在线批量规划格式。每个批量样本必须包含全部请求 agent，顺序和 ID 均可验证。

### 9.3 DPO 数据

```json
{
  "example_id": "...",
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "<better JSON>"}],
  "rejected": [{"role": "assistant", "content": "<worse JSON>"}],
  "preference": {
    "chosen_score": 0.91,
    "rejected_score": 0.63,
    "score_gap": 0.28,
    "scorer_version": "plan_scorer_v1",
    "candidate_sources": ["teacher", "rule_perturbation"]
  }
}
```

偏好对要求：

- `chosen` 和 `rejected` 都尽量满足 JSON/schema 基本格式，使模型学习决策优劣而不只是格式优劣。
- 分差必须高于配置阈值；低置信度对进入隔离区，不直接训练。
- 完全不可达、缺失 agent 或越权访问区域可作为少量格式负例，但不能占据多数。
- 同一个 prompt 的候选只能进入同一数据分区。

## 10. 从真实轨迹生成标签

### 10.1 事实提取

对每条合格轨迹提取：

- 起点、终点、持续时间、路径长度、平均/分位速度。
- 区域访问序列、首次进入时间、停留时间、回访次数。
- 入口/出口映射和无法匹配标记。
- 低速、驻留、拥堵暴露等使用统一指标定义的特征。
- 可用于画像的历史特征；目标时间窗之后的信息不得进入输入。

### 10.2 真值投影

真实轨迹并不天然等于模型输出 JSON，需要确定性投影器：

```text
continuous trajectory
  -> coordinate transform
  -> region membership sequence
  -> temporal de-duplication
  -> valid route graph projection
  -> activity/dwell labels
  -> final exit
  -> llm_route_response JSON
```

投影失败不能静默修补。记录失败原因，如 `outside_map`、`ambiguous_region`、`unreachable_transition`、`missing_exit`，并在数据报告中展示占比。

### 10.3 教师候选生成

教师 API 的职责是为 train/tuning 样本补充多个可比较候选，不直接参与最终仿真条件。推荐每个 prompt 生成：

- 1 个真值投影候选（若有效）。
- 2 至 4 个教师模型候选，使用不同 seed/temperature。
- 1 个规则规划器候选。
- 1 至 2 个受控扰动候选，例如次优出口、过长路径或画像不匹配行为。

需要新增缓存，键至少包含：

```text
teacher_provider + teacher_model + prompt_hash + generation_config_hash
```

教师调用日志记录 token、延迟、重试、错误和成本估计，但不记录 API key。所有响应先落原始缓存，再做解析，以便评分器变更时无需重复付费调用。

### 10.4 候选验证和评分

先执行硬校验：

- JSON 可解析且符合 response schema。
- agent 集合完整且无重复。
- 区域和出口存在。
- 路线图上可达。
- 速度、活动、停留时间等数值在合理范围。

再执行版本化软评分。`plan_scorer_v1` 建议包含：

```text
total =
  w_route    * route_similarity_or_efficiency
  + w_exit   * exit_accuracy
  + w_dwell  * dwell_plausibility
  + w_profile* profile_alignment
  + w_flow   * congestion_or_flow_objective
  + w_format * completeness
```

权重只能用 train/tuning 确定。评分报告必须给出每个分量，不能只保存总分。对于无法从真实数据证明的主观画像偏好，应标记为教师/规则假设，论文中不能表述为真实观测事实。

### 10.5 人工抽检

正式训练前至少抽检：

- 随机 SFT 样本。
- 分差最小和最大的 DPO 偏好对。
- 每类失败原因样本。
- 每个画像、入口、出口和主要区域组合。

抽检结论和修改记录写入 dataset card。

## 11. SFT 训练方案

### 11.1 基础模型选择

第一版建议使用可本地部署、许可允许研究使用、具有稳定 chat template 的 Qwen 7B/8B 级指令模型。模型必须固定：

- 仓库 ID 或本地路径。
- 精确 revision/commit。
- tokenizer revision。
- 许可版本。
- 下载文件哈希或 Hugging Face snapshot ID。

不要在训练脚本中写死某个 Windows/Linux 路径，也不要出现训练使用 8B、推理却默认加载 4B 的模型错配。

### 11.2 QLoRA 初始配置

建议作为显存实测起点，而不是未经实验的最终论文参数：

```yaml
method: qlora_sft
quantization:
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
training:
  learning_rate: 0.0002
  epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  max_sequence_length: 2048
  gradient_checkpointing: true
  seed: 20260715
```

服务器如果不支持 BF16，显式改为 FP16 并记录。先用 32 至 128 个样本完成 overfit/smoke，确认 loss 下降、JSON 输出可解析、checkpoint 可恢复，再运行全量训练。

### 11.3 SFT 验收

- train loss 和 tuning loss 有记录。
- tuning 集 JSON/schema 合法率、agent 完整率、可达率显著高于未训练基线，或至少无退化。
- 固定 prompt 的生成结果可复现到统计意义上的一致。
- adapter 能单独加载，且 base revision 与训练时一致。
- `training_run_manifest.json` 完整并通过 schema 校验。

## 12. DPO 训练方案

DPO 从已验收的 SFT adapter 继续训练。初始配置：

```yaml
method: qlora_dpo
base_adapter: mall_sft_v1
learning_rate: 0.000005
beta: 0.1
epochs: 1
per_device_batch_size: 1
gradient_accumulation_steps: 32
max_prompt_length: 1024
max_sequence_length: 2048
seed: 20260715
```

实施要求：

- SFT 和 DPO 可以来自同一 train 人员分区，但 DPO tuning 必须独立于训练样本。
- DPO loader 必须验证 chosen/rejected 均匹配当前 chat template。
- 记录 reward chosen、reward rejected、reward margin、KL/隐式偏移等训练指标。
- 比较 SFT 与 SFT+DPO，不允许只报告 DPO 后模型。
- 若 DPO 提高离线评分却降低 schema 合法率或仿真稳定性，应回退并调整偏好数据，而不是仅调大训练轮数。

第一阶段不引入 GRPO/PPO。只有在可执行、确定性、不会奖励投机的仿真 reward 建立后，才将其列为扩展实验。

## 13. 训练运行清单与断点恢复

每次训练生成独立目录：

```text
checkpoints/<run_id>/
  resolved_training_config.yaml
  training_run_manifest.json
  dataset_manifest.json
  environment.txt
  pip_freeze.txt
  git_diff.patch
  logs/
  checkpoint-*/
  eval/
```

`training_run_manifest.json` 至少记录：

- `run_id`、开始/结束时间、状态。
- Git commit、dirty worktree 标记和必要的 diff 哈希。
- 数据集 ID、分区哈希、样本数。
- 基础模型、tokenizer 和 adapter revision。
- 完整超参数、seed、prompt/schema/scorer 版本。
- GPU 型号、数量、驱动、CUDA、PyTorch、Transformers、PEFT、TRL 版本。
- 最佳 checkpoint、选择指标和 tuning 结果。
- 父训练 run，例如 DPO 指向 SFT run。

训练命令支持 `--resume-from-checkpoint`。写数据集和发布 adapter 时使用临时目录加原子重命名，避免中断后出现“看似完整”的半成品。

长期任务使用 Slurm、systemd user service 或 tmux；标准输出同时写日志文件。不得仅依赖 SSH 会话存活。

## 14. 模型发布与 API 服务

### 14.1 模型注册表

每个可用于实验的模型建立只读登记项：

```yaml
model_id: mall-qwen8b-sft-dpo-v1
base_model: /srv/models/<base-snapshot>
base_revision: <revision>
adapter_path: /srv/jupedsim-artifacts/adapters/mall-qwen8b-sft-dpo-v1
adapter_sha256: <manifest hash>
training_run_id: <run id>
served_model_name: mall-qwen8b-sft-dpo-v1
prompt_version: "1.0"
response_schema_version: "1.0"
status: candidate
```

状态建议为 `candidate -> validated -> frozen -> retired`。只有 `frozen` 模型进入正式论文矩阵。

### 14.2 OpenAI-compatible 服务

优先使用 vLLM 或 SGLang 加载基础模型和 LoRA adapter。以实际安装版本文档为准，示意命令：

```bash
source /srv/jupedsim-mall/.venv-serve/bin/activate
vllm serve /srv/models/<base-snapshot> \
  --host 127.0.0.1 \
  --port 8600 \
  --served-model-name mall-base \
  --enable-lora \
  --lora-modules mall-sft-dpo=/srv/jupedsim-artifacts/adapters/<adapter>
```

若仿真和服务不在同一主机，通过防火墙受控内网地址访问；不要将无认证服务暴露到公网。

健康检查：

```bash
curl -s http://127.0.0.1:8600/v1/models
curl -s http://127.0.0.1:8600/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/llm_health_request.json
```

服务验收应包含 schema 合法率、并发、超时、重试、吞吐、GPU 显存和固定测试集输出。服务版本、模型名和 adapter 哈希写入每个仿真 run manifest。

### 14.3 仿真侧接入

```bash
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8600/v1
export LOCAL_LLM_MODEL=mall-sft-dpo
export LOCAL_LLM_API_KEY=local-only-placeholder
```

当前场景中固定的 URL/模型名必须先按第 7 节移除，否则场景环境会覆盖服务器设置。仿真失败时记录实际 HTTP 状态、重试和 fallback；不能静默把规则 fallback 结果计为 LLM 成功结果。

## 15. 论文实验矩阵

### 15.1 必须保留的模型条件

| 条件 | 含义 | 目的 |
|---|---|---|
| `B1_RULE` | 原始规则基线 | 验证 LLM 是否必要 |
| `B2_PROFILE_RULE` | 使用画像但无 LLM | 区分画像贡献 |
| `M0_BASE_PROMPT` | 未训练基础模型 + 当前提示词 | 测量提示工程基线 |
| `M1_SFT` | QLoRA-SFT | 测量监督微调贡献 |
| `M2_SFT_DPO` | SFT 后 DPO | 测量偏好优化贡献 |
| `A_NO_PROFILE` | 训练模型但移除画像 | 画像消融 |
| `A_NO_DPO` | 等同 SFT 模型 | DPO 消融 |
| `P_ORACLE_OR_REPLAY` | 真值投影/回放上界，若场景支持 | 给出可达上界，不与生成模型混淆 |

可继续保留现有 B1/B2/A1/M1/P1/P2 条件，但应映射到清晰的新命名，并在报告中说明旧 ID。模型条件、场景条件、数据集条件不要混在一个模糊字段中。

### 15.2 配对设计

- 所有方法使用相同场景、人员样本、仿真 seed 和到达过程。
- LLM 解码 seed、temperature、top-p 固定并记录。
- 首先执行小规模 pilot，冻结参数后再执行 formal。
- 重复次数由功效分析或预实验方差决定，不能只因运行成本任意选择。
- 失败运行按预注册规则处理；不得只删除不利于某模型的失败。

### 15.3 离线模型指标

- JSON/schema 合法率。
- agent 完整率和 ID 正确率。
- 路线可达率、出口准确率。
- 区域序列 Jaccard、编辑距离或 LCS。
- 停留时间和速度误差。
- 画像一致性评分及其人工抽检一致率。
- 推理延迟、吞吐、token 数和显存。
- fallback 率、超时率、重试率。

### 15.4 仿真系统指标

- 到达/疏散成功率。
- 平均路径长度和旅行时间。
- 拥堵、密度、低速比例、停留和区域负载。
- 不同画像/入口/出口子组差异。
- 与 evaluation 真实参考分布的距离。
- 方法间成对差、置信区间、效应量和多重比较校正。

离线指标好不等于仿真效果好，因此模型发布和论文结论必须同时报告两层结果。

## 16. Linux 端到端命令设计

以下命令是施工后应达到的接口，当前并非全部已经存在：

```bash
# 1. 环境与数据检查
./scripts/run_doctor.sh --mode training

# 2. 生成冻结后的数据事实与分区
python scripts/prepare_real_data.py \
  --config configs/data/atc_dataset_v1.yaml \
  --output /srv/jupedsim-data/processed/atc_dataset_v1

# 3. 构建真值、规则候选和教师请求
python scripts/build_training_dataset.py \
  --config configs/data/atc_dataset_v1.yaml \
  --stage seed

# 4. 调用教师 API，并可安全断点续跑
python scripts/generate_teacher_candidates.py \
  --dataset-id atc_plan_v1 \
  --resume

# 5. 校验、评分并导出 SFT/DPO
python scripts/build_training_dataset.py \
  --config configs/data/atc_dataset_v1.yaml \
  --stage finalize

# 6. 泄漏和数据质量门禁
python -m jupedsim_mall.training.leakage_audit \
  --dataset /srv/jupedsim-artifacts/datasets/atc_plan_v1

# 7. SFT 冒烟与正式训练
python scripts/train_sft.py \
  --config configs/training/qwen_8b_qlora_sft_v1.yaml \
  --max-samples 64
python scripts/train_sft.py \
  --config configs/training/qwen_8b_qlora_sft_v1.yaml

# 8. DPO
python scripts/train_dpo.py \
  --config configs/training/qwen_8b_qlora_dpo_v1.yaml

# 9. 离线评估和模型登记
python scripts/evaluate_trained_model.py --model-ref mall_sft_dpo_v1

# 10. 启动模型服务
./scripts/serve_model.sh configs/models/mall_sft_dpo_v1.yaml

# 11. pilot 与 formal 仿真
./scripts/run_experiment_suite.sh configs/experiments/pilot_trained_models.yaml
./scripts/run_experiment_suite.sh configs/experiments/formal_trained_models.yaml

# 12. 全工程质量门禁
./scripts/run_quality_gate.sh
```

## 17. 分阶段施工清单

### L0：冻结当前研究状态

- [ ] 保存当前 Git commit、分支、未提交 diff 和现有测试结果。
- [ ] 将当前报告标记为 `engineering_smoke`。
- [ ] 保存现有配置、输入文件和关键 provenance 哈希。
- [ ] 创建迁移标签，例如 `pre-linux-training-v1`。

完成标准：在新机器上可以还原“训练侧施工前”的工程状态。

### L1：Linux 仿真基线

- [ ] 建立目录、权限、`.venv-sim` 和 Linux lock。
- [ ] 增加 `.sh` 入口并通过 LF/可执行位检查。
- [ ] 移除 Windows 绝对路径假设。
- [ ] 在 Linux 跑完单元测试、doctor 和最小 JuPedSim 场景。
- [ ] 对比 Windows/Linux 固定 seed 的关键指标，解释浮点或依赖差异。

完成标准：未引入训练时，Linux 已能稳定复现当前工程流水线。

### L2：数据契约与真实性门禁

- [ ] 实现坐标变换配置和轨迹叠加检查。
- [ ] 统一速度/驻留指标版本。
- [ ] 修正抽样 seed 语义。
- [ ] 固化人员级 train/tuning/evaluation 切分。
- [ ] 增加泄漏审计和数据 manifest schema。

完成标准：真实数据的每个派生值都能追溯到原始文件、配置和代码版本。

### L3：训练数据构建

- [ ] 实现轨迹事实提取和真值投影。
- [ ] 复用现有输出 schema 和可达性验证。
- [ ] 实现教师候选缓存、重试和成本日志。
- [ ] 实现版本化候选评分器和 hard-negative 选择。
- [ ] 导出 SFT/DPO JSONL、dataset card 和质量报告。
- [ ] 完成人工抽检。

完成标准：数据集 schema、泄漏、分差、覆盖率和人工抽检门禁全部通过。

### L4：QLoRA-SFT

- [ ] 固定基础模型 revision 和许可。
- [ ] 完成 32 至 128 样本的过拟合/显存冒烟。
- [ ] 实现 checkpoint 恢复和训练 manifest。
- [ ] 执行全量 SFT，并只用 tuning 选 checkpoint。
- [ ] 完成离线输出格式与路线质量评估。

完成标准：SFT adapter 可独立加载，离线测试和运行清单完整。

### L5：DPO

- [ ] 从验收后的 SFT adapter 初始化。
- [ ] 验证偏好对、chat template 和 token 长度。
- [ ] 训练并记录 reward margin 等指标。
- [ ] 对比 BASE、SFT、SFT+DPO tuning 表现。
- [ ] 检查 DPO 是否导致格式合法率或可达率退化。

完成标准：DPO adapter 通过预定义提升门槛；若未通过，保留负结果并以 SFT 作为候选模型。

### L6：服务化和仿真接入

- [ ] 建立模型注册表和不可变 adapter 发布目录。
- [ ] 部署 vLLM/SGLang OpenAI-compatible API。
- [ ] 修改场景环境优先级，移除固定 URL/模型名。
- [ ] 记录实际服务模型、adapter 哈希、延迟和 fallback。
- [ ] 完成并发、超时和重启恢复测试。

完成标准：现有 `LLMRoutePlanner` 无需加载模型权重即可稳定调用 BASE/SFT/DPO。

### L7：pilot 实验

- [ ] 建立 BASE、SFT、SFT+DPO、规则和画像消融矩阵。
- [ ] 使用 tuning 或独立 pilot 数据调试，不触碰 evaluation 结论。
- [ ] 估计重复次数、运行时间、失败率和磁盘需求。
- [ ] 冻结场景、模型、指标、统计方法和排除规则。

完成标准：生成 formal protocol 和冻结配置哈希。

### L8：formal 实验与论文报告

- [ ] 在 evaluation 分区运行全部冻结条件和配对 seed。
- [ ] 运行完整性、失败和 provenance 审计。
- [ ] 生成离线模型指标与仿真系统指标。
- [ ] 执行统计检验、效应量和置信区间分析。
- [ ] 报告负结果、fallback 和资源成本。

完成标准：报告中的每张表和图都可回溯到 run IDs，且没有训练/评估泄漏。

### L9：归档与续研

- [ ] 冻结代码 tag、配置、manifest、adapter 和报告。
- [ ] 生成环境重建说明和模型/数据 cards。
- [ ] 记录未完成假设、失败实验和下一轮变量。
- [ ] 对受限原始数据只保存哈希和重建说明，不违规打包。

完成标准：更换服务器或人员后，能够根据归档继续实验。

## 18. 资源规划与降级路线

在不知道服务器 GPU 型号前，不承诺具体可训练 batch size。7B/8B QLoRA 通常适合从单张中高显存 GPU 试起，但序列长度、attention 实现、batch 和库版本都会显著改变显存占用。

拿到服务器后先做显存探测：

```text
64 samples -> max_length 1024 -> batch 1 -> grad accumulation
逐步提升 max_length 到数据 P95 所需长度
记录峰值显存、tokens/s 和 checkpoint 时间
```

降级顺序：

1. 降低在线/训练 batch size。
2. 降低最大序列长度，但必须报告截断率。
3. 启用 gradient checkpointing 和高效 attention。
4. 调整 LoRA target/rank。
5. 换用更小基础模型，并作为不同模型条件重新登记。

禁止因显存不足静默截断输出或把不同基础模型的结果仍标为同一实验条件。

## 19. 密钥、安全和隐私

新增 `.env.example`，只包含变量名：

```dotenv
TEACHER_LLM_BASE_URL=
TEACHER_LLM_MODEL=
TEACHER_LLM_API_KEY=
LOCAL_LLM_BASE_URL=http://127.0.0.1:8600/v1
LOCAL_LLM_MODEL=
LOCAL_LLM_API_KEY=
```

真实 `.env`、shell history 中的密钥、教师原始响应中的敏感字段都不得提交。日志层实现 key/header 脱敏。外部模型供应商的数据保留和训练政策应在上传真实轨迹上下文之前核对；必要时只发送匿名化、离散化区域序列。

## 20. 常见失败及判定

| 现象 | 首要检查 | 处理原则 |
|---|---|---|
| Linux 测试与 Windows 不一致 | 依赖版本、路径大小写、seed、浮点库 | 先复现基线，不直接进入训练 |
| 训练 loss 降低但 JSON 失败 | chat template、label mask、截断 | 修复契约并重建数据 |
| SFT 输出漏 agent | 训练 batch 格式与在线 batch 不一致 | 第一版降为单 agent，后续加混合批量训练 |
| DPO 只学会格式差异 | rejected 多为非法 JSON | 增加合法但决策较差的 hard negatives |
| 离线分数提高但仿真变差 | 评分器与系统目标错位 | 以成对仿真验证，修订评分器版本 |
| LLM 条件结果异常接近规则 | fallback 被静默使用 | 报告 fallback 并将高 fallback run 判失败 |
| evaluation 表现被反复调参改善 | 数据泄漏 | 作废该轮 formal，重新冻结协议 |
| API 模型与训练模型不一致 | served name、base revision、adapter | 由模型注册表和健康检查阻断运行 |

## 21. 每轮研究续接记录模板

每轮施工或实验结束后，在 `docs/research_progress/` 新建记录：

```markdown
# Research progress: <date>-<topic>

## Objective

## Code state
- Git commit:
- Branch:
- Dirty diff hash:

## Data state
- Dataset ID:
- Raw data hashes:
- Split hash:
- Dataset manifest:

## Model state
- Base model/revision:
- SFT run/adapter:
- DPO run/adapter:
- Registry status:

## Experiments completed
- Run IDs:
- Reports:
- Key result:

## Decisions and reasons

## Failed attempts

## Open risks

## Exact next command
```

该记录不是替代机器生成 manifest，而是帮助后续研究者快速理解“为什么这样做”和“下一步从哪里继续”。

## 22. 开工前待确认参数

以下参数应在 Linux 服务器信息可用后填写，不应凭空假设：

| 参数 | 待填写值 |
|---|---|
| Linux 发行版 |  |
| GPU 型号、数量、显存 |  |
| NVIDIA driver / CUDA runtime |  |
| CPU、RAM、可用磁盘 |  |
| 调度方式（直接/tmux/Slurm） |  |
| 基础模型 ID 与 revision |  |
| 教师模型供应商、模型、预算 |  |
| ATC 数据许可和服务器存放路径 |  |
| LLMob 数据许可和服务器存放路径 |  |
| 坐标变换确认人和版本 |  |
| 正式实验最大运行预算 |  |

## 23. 最终完成定义

只有同时满足以下条件，才可以声明“完整训练实验流水线已完成”：

- Linux 上可从干净环境重建仿真、训练和服务环境。
- 原始数据哈希、坐标标定、指标定义和人员切分已冻结。
- SFT/DPO 数据通过 schema、质量、人工抽检和泄漏门禁。
- BASE、SFT、SFT+DPO 模型的 revision、adapter 和训练过程可追溯。
- 本地模型以 OpenAI-compatible API 接入，fallback 和失败不会被隐瞒。
- pilot 与 formal 数据严格分离，formal 配置在运行前冻结。
- 论文报告同时包含离线模型指标、仿真指标、统计不确定性和资源成本。
- 图表、表格、模型和数据结论均能回溯到不可变 run manifest。

在这些条件完成之前，工程可以称为“训练侧开发中”或“工程验收通过”，不能称为“真实数据训练后的正式论文实验已完成”。

## 24. 方法依据

- QLoRA：<https://arxiv.org/abs/2305.14314>
- Direct Preference Optimization：<https://arxiv.org/abs/2305.18290>
- Hugging Face TRL SFT Trainer：<https://huggingface.co/docs/trl/en/sft_trainer>
- Hugging Face TRL PEFT integration：<https://huggingface.co/docs/trl/peft_integration>
- Hugging Face TRL DPO Trainer：<https://huggingface.co/docs/trl/en/dpo_trainer>
- Qwen training / Unsloth documentation：<https://qwen.readthedocs.io/en/latest/training/unsloth.html>

这些资料用于确定通用方法。实际版本、参数和命令必须以服务器安装时的官方文档、锁定依赖和本工程实测为准。
