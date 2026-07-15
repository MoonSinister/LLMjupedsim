# 毕业设计工程流程与功能设计

本文档用于把当前项目从“可运行演示”整理为“可复现、可对比、可写入论文”的毕业设计工程。组织方式分为两条主线：

- **流程线**：从地图数据到实验结果的完整研究流水线。
- **功能线**：工程中各模块分别承担什么职责、输入输出是什么、后续如何扩展。

## 一、研究定位

本项目围绕“基于语义地图与大语言模型的室内行人仿真”展开，核心思想是：用 JuPedSim 负责物理行人动力学，用语义地图描述室内区域含义，用 LLM 为不同画像的行人生成高层行为路径，再通过实验流水线比较不同策略下的人群运动结果。

可以在论文中定位为四层融合：

1. **物理仿真层**：JuPedSim 提供可复现的行人运动、出口、阶段、轨迹记录能力。
2. **语义地图层**：地图不仅包含可行走区域，还包含出口、入口、店铺、餐厅、扶梯、走廊等语义区域。
3. **行为规划层**：LLM 根据行人画像、出生区域、候选出口和语义区域生成意图、活动点、等待时间和最终出口。
4. **实验评估层**：通过 baseline、LLM 路由、ATC/LLMob 画像等实验组进行对比，并输出论文表格和可视化材料。

可引用或参考的方向：

- Park et al., “Generative Agents: Interactive Simulacra of Human Behavior”, 2023：用于说明自然语言画像和生成式智能体的研究背景。https://arxiv.org/abs/2304.03442
- JuPedSim 官方工程与文档：用于说明物理仿真引擎基础。https://www.jupedsim.org/
- Agent-based simulation / digital twin 实验范式：固定地图和物理仿真器，比较不同路径规划策略、画像来源和参数消融。

## 二、整体流程

### 1. 地图构建流程

目标：把原始地图转化为 JuPedSim 可用的几何和语义输入。

输入文件：

- `data/map/localization_grid.pgm`
- `data/map/localization_grid.yaml`
- `data/map/drawn_geometry.json`
- `data/map/geometry.wkt`
- `data/map/stages.json`
- `data/map/localization_grid_regions.json`

相关工具：

- `src/manual_draw_geometry.py`：交互式绘制可行走区域、墙体、出口、入口。
- `src/geometry_editor.py`：几何体查看、裁剪、平滑、孔洞处理等辅助编辑。
- `src/pgm_to_geometry.py`：从 PGM 栅格地图提取初始几何。
- `src/plot_free_points.py`：查看自由空间点和辅助标注区域。

输出结果：

- `geometry.wkt`：最终可行走区域。
- `stages.json`：出口和入口区域。
- `localization_grid_regions.json`：语义区域。

质量要求：

- 出口和入口必须与可行走区域相交。
- 语义区域应尽量覆盖重要功能空间。
- 几何编辑应保留 `drawn_geometry.json`，保证后续可追溯和可修改。

### 2. 行人画像构建流程

目标：为不同 agent 提供异质化行为基础，使仿真不是同质随机行人。

画像来源：

- `mall`：内置商场行人画像，包括 commuter、shopper、staff、visitor。
- `llmob`：从 LLMob 数据中抽取日常活动规律，映射为室内行人画像。
- `atc`：从 ATC 轨迹数据中识别速度、停留、访问区域等行为特征。

相关模块：

- `src/llmob_adapter.py`
- `src/llmob_training_adapter.py`
- `src/agent_model.py`

输出结果：

- `outputs/profiles/*.json`

论文中可对比的问题：

- 内置画像与数据驱动画像是否产生不同的角色分布。
- 数据驱动画像是否提升行为解释性。
- 不同画像来源是否影响拥堵、停留、出口分布和 TTL 移除。

### 3. LLM 路径规划流程

目标：让行人拥有可解释的高层行为计划，而不是只随机选择出口。

输入信息：

- agent 画像。
- 出生区域。
- 候选出口。
- 语义区域名称和描述。

规划输出：

- `role`：行人角色。
- `subtype`：细分类型。
- `intent`：自然语言意图。
- `activities`：中间活动区域、行为名称、等待时间。
- `final_exit`：最终出口。
- `desired_speed_mps`：期望速度。

相关模块：

- `src/llm_prompts.py`
- `src/demo_map_simulation.py` 中的 `route_agents_with_llm`

输出结果：

- `outputs/plans/*.json`

质量要求：

- LLM 只能选择候选出口中的出口。
- LLM 只能选择存在于语义区域文件中的区域。
- LLM 失败时必须回退到 baseline 路由，避免实验中断。
- 每个 agent 的计划应能用于论文中的可解释案例分析。

### 4. 物理仿真流程

目标：把高层行为计划转化为 JuPedSim journey/stage，并生成轨迹。

主入口：

- `src/demo_map_simulation.py`

关键能力：

- 加载地图几何和手工 stages。
- 创建出口、入口、agent 队列。
- 支持 random baseline 和 LLM route。
- 支持语义 waypoint。
- 支持等待行为。
- 支持拥堵检测、绕行、TTL 移除。
- 输出 SQLite 轨迹。

输出结果：

- `outputs/trajectories/*.sqlite`
- `outputs/smoke/*.sqlite`

### 5. 实验运行流程

目标：把实验从手工命令升级为可复现的场景配置。

场景配置目录：

- `configs/scenarios/smoke_baseline.json`
- `configs/scenarios/baseline_random.json`
- `configs/scenarios/llm_remote.json`
- `configs/scenarios/atc_llm_remote.json`

运行入口：

- `src/run_experiment_suite.py`
- `scripts/run_scenario.ps1`

常用命令：

```powershell
.\scripts\run_scenario.ps1 --list
.\scripts\run_scenario.ps1 --dry-run baseline_random llm_remote
.\scripts\run_scenario.ps1 smoke_baseline
.\scripts\run_scenario.ps1 baseline_random llm_remote
```

输出结果：

- `outputs/runs/experiment_manifest_*.json`

manifest 的意义：

- 记录每次实验使用的场景配置。
- 记录实际执行命令。
- 记录开始和结束时间。
- 为论文复现实验提供证据。

### 6. 指标汇总流程

目标：把 plan、profile、trajectory 结果汇总为论文表格材料。

运行入口：

- `src/experiment_summary.py`

常用命令：

```powershell
D:\anaconda\envs\python3.12\python.exe src\experiment_summary.py
```

输出结果：

- `outputs/summaries/experiment_summary.json`
- `outputs/summaries/experiment_summary.csv`

当前已支持指标：

- plan 文件中的 agent 数量。
- 角色分布。
- 出口分布。
- 平均中间活动数量。
- TTL 移除数量。
- 拥堵重路由数量。
- profile 文件中的画像数量和角色分布。
- SQLite 轨迹文件中的表行数统计。

建议后续扩展指标：

- 总疏散时间。
- 单个 agent 的平均旅行时间。
- 最大旅行时间。
- 出口利用熵。
- 区域密度峰值。
- 瓶颈区域停留时间。
- LLM 成功率和 fallback 比例。

## 三、功能模块设计

### 1. 地图与几何模块

职责：

- 管理可行走区域、障碍物、出口、入口、语义区域。
- 保证地图数据可以编辑、保存、复现。

主要文件：

- `src/manual_draw_geometry.py`
- `src/geometry_editor.py`
- `src/pgm_to_geometry.py`
- `data/map/*`

后续建议：

- 抽取 `geometry_io.py`，统一 WKT、JSON、Python geometry 的读写。
- 给 `stages.json` 和 `localization_grid_regions.json` 增加 schema 校验。

### 2. 画像模块

职责：

- 为 agent 生成角色、偏好、速度、等待时间、个体差异。
- 支持 synthetic 和 data-driven 两类来源。

主要文件：

- `src/agent_model.py`
- `src/llmob_adapter.py`
- `src/llmob_training_adapter.py`

后续建议：

- 增加 `AgentProfile` dataclass。
- 为不同画像来源输出统一字段。
- 增加画像分布统计图。

### 3. LLM 规划模块

职责：

- 构建 prompt。
- 调用 OpenAI-compatible LLM 服务。
- 解析并校验 LLM route。
- 失败时提供 fallback。

主要文件：

- `src/llm_prompts.py`
- `src/demo_map_simulation.py`

后续建议：

- 抽取 `llm_router.py`。
- 增加 JSON schema 校验。
- 记录每个 batch 的成功率、失败原因、token 预算。
- 支持离线 mock route，方便无 LLM 环境测试。

### 4. 仿真执行模块

职责：

- 调用 JuPedSim。
- 创建 journey、stage、agent。
- 执行主循环。
- 处理拥堵、等待、TTL 和输出。

主要文件：

- `src/demo_map_simulation.py`

后续建议：

- 拆分为 `simulation_runner.py`、`congestion_control.py`、`journey_builder.py`。
- 将主循环中的状态 dict 替换为 dataclass。
- 增加小规模自动化 smoke test。

### 5. 实验管理模块

职责：

- 用配置文件描述实验。
- 批量运行多个场景。
- 生成 manifest。

主要文件：

- `configs/scenarios/*.json`
- `src/run_experiment_suite.py`
- `scripts/run_scenario.ps1`

后续建议：

- 为 scenario config 增加 schema。
- 增加随机种子字段。
- 支持重复运行 N 次并自动编号输出目录。

### 6. 指标分析模块

职责：

- 从实验产物中提取统计指标。
- 输出 JSON/CSV，服务论文表格和画图。

主要文件：

- `src/experiment_summary.py`

后续建议：

- 读取轨迹数据库，计算真实旅行时间和完成率。
- 增加 Matplotlib 图表输出。
- 支持按场景自动对齐比较。

## 四、实验矩阵

毕业设计建议至少包含以下实验：

| 场景 | 画像来源 | 路由策略 | 作用 |
| --- | --- | --- | --- |
| `smoke_baseline` | mall | random | 快速验证工程是否能跑通 |
| `baseline_random` | mall | random | 对照组 |
| `llm_remote` | mall | LLM semantic route | 主方法 |
| `atc_llm_remote` | ATC | LLM semantic route | 数据驱动画像扩展 |

建议消融实验：

- 关闭 routing waypoints。
- 改变 `--waypoint-distance`。
- 改变 `--max-active-agents`。
- 对比 `avm` 和 `cfsv3`。
- 对比 `mall`、`llmob`、`atc` 三类画像。
- 对比不同 LLM batch size 和 max token budget。

## 五、论文结构对应关系

### 第 1 章：绪论

说明室内行人仿真的应用背景，指出传统随机路由缺少语义行为解释的问题，引出 LLM + 语义地图 + 物理仿真的研究目标。

### 第 2 章：相关工作

讨论行人动力学、agent-based simulation、生成式智能体、LLM 行为规划、移动轨迹画像建模。

### 第 3 章：系统设计

介绍地图层、画像层、LLM 规划层、JuPedSim 仿真层、实验管理层和指标分析层。

### 第 4 章：系统实现

介绍几何编辑工具、语义区域标注、prompt schema、fallback 路由、拥堵处理、TTL 控制、场景配置和批量运行。

### 第 5 章：实验与分析

展示 baseline、LLM route、ATC/LLMob profile、参数消融实验，分析出口分布、角色分布、拥堵重路由、TTL、轨迹可视化。

### 第 6 章：总结与展望

总结方法优势与局限，提出在线重规划、真实密度指标校准、更强 schema 校验、多地图泛化等后续方向。

## 六、当前已完成与下一步

当前已完成：

- 场景配置目录 `configs/scenarios`。
- 批量运行器 `src/run_experiment_suite.py`。
- PowerShell 运行入口 `scripts/run_scenario.ps1`。
- 指标汇总脚本 `src/experiment_summary.py`。
- 毕业设计流程文档。
- 实验重复运行、随机种子控制和 manifest 可追溯记录。
- baseline route、nearest route、LLM exit-only、LLM semantic route 等对照组。
- plan validation，用于检查出口、候选出口、语义区域、等待时间和速度范围是否合法。
- aggregate summary，用于对重复实验输出均值和标准差。

下一步优先级：

1. 抽取 `AgentProfile`、`AgentRoute`、`SpawnPlan`、`SemanticRegion` dataclass。
2. 为 LLM 输出增加更严格的 JSON schema 校验。
3. 从 SQLite 轨迹中进一步计算完成率、区域密度和瓶颈停留时间。
4. 增加 `tests/`，覆盖配置加载、路线校验、几何加载和指标汇总。
5. 增加图表生成脚本，输出论文可直接使用的柱状图、折线图和热力图。

## 七、实验完备性增强

为了让论文实验更严谨，工程新增以下机制：

### 1. 可重复实验

每个实验场景都可以通过 `--repeat` 重复运行，并通过 `--seed-start` 控制随机种子序列：

```powershell
.\scripts\run_scenario.ps1 --repeat 5 --seed-start 2026 baseline_random baseline_nearest llm_exit_only llm_remote
```

该命令会对每个场景运行 5 次，种子依次为 2026、2027、2028、2029、2030。重复实验的输出文件会自动添加 `seedXXXX_runXX` 后缀，避免覆盖。

### 2. 对照组与消融组

当前建议使用以下实验矩阵：

| 场景 | 路由策略 | 论文作用 |
| --- | --- | --- |
| `baseline_random` | 随机候选出口 | 最基础对照组 |
| `baseline_nearest` | 最近候选出口 | 几何最短倾向对照组 |
| `llm_exit_only` | LLM 只选择最终出口 | 检验 LLM 出口选择能力 |
| `llm_remote` | LLM 语义活动点 + 最终出口 | 主方法 |
| `atc_llm_remote` | ATC 画像 + LLM 语义路径 | 数据驱动画像扩展 |

这样可以分别回答：

- LLM 是否优于随机策略。
- LLM 是否优于简单最近出口策略。
- 语义中间活动点是否带来额外效果。
- 数据驱动画像是否增强行为异质性。

### 3. 路线计划校验

所有 plan 文件都会包含 `validation` 字段。校验内容包括：

- `exit_idx` 是否存在。
- 最终出口是否属于该出生区域的候选出口。
- 中间语义区域是否存在于地图语义区域文件中。
- `wait_seconds` 是否在允许范围内。
- `desired_speed_mps` 是否在允许范围内。

这些校验项可以直接用于论文中的“有效性检查”或“实验质量控制”小节。

### 4. 可追溯 Manifest

批量运行器会在 `outputs/runs/experiment_manifest_*.json` 中记录：

- 场景名称。
- repeat 编号。
- seed。
- 实际执行命令。
- 场景配置快照。
- git commit。
- Python 版本。
- 地图关键文件的 SHA256 hash。
- 开始和结束时间。

这使实验具备可复现性证据，避免论文结果只停留在口头描述。

### 5. 聚合指标

`src/experiment_summary.py` 会输出：

- `outputs/summaries/experiment_summary.json`
- `outputs/summaries/experiment_summary.csv`
- `outputs/summaries/experiment_summary_aggregate.json`
- `outputs/summaries/experiment_summary_aggregate.csv`

其中 aggregate 文件按场景聚合重复实验，输出均值和标准差。当前支持：

- agent 数。
- TTL 移除率。
- 拥堵重路由率。
- 平均语义活动数量。
- plan validation 错误数量。
- trajectory 观测人数。
- trajectory 持续时间。
- 平均观测旅行时间。
- 最大观测旅行时间。
