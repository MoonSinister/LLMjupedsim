# JuPedSim Mall Simulation 工程化技术路线

本文档定义当前项目从“可运行实验脚本”演进为“可复现、可扩展、可论文交付”的工程路线。重构原则是：先稳定数据契约和实验接口，再逐步拆分 `demo_map_simulation.py`，不在一次改动中改变仿真语义。

## 1. 项目定位

本项目研究“语义地图 + LLM 行为规划 + JuPedSim 物理仿真”的室内行人仿真方法。

技术分工如下：

- JuPedSim：负责连续空间中的行人动力学、碰撞规避、出口、阶段和轨迹记录。
- 语义地图：把几何空间扩展为入口、出口、店铺、餐饮、楼梯、走廊等可解释区域。
- Agent profile：为行人提供角色、偏好、速度、停留倾向和历史行为线索。
- LLM route planner：把 profile 和语义地图转化为高层活动计划。
- 实验系统：用 scenario、seed、manifest、summary 保证结果可复现、可对比。

## 2. 目标架构

目标模块边界：

```text
src/
|-- sim_schema.py              # 稳定数据模型和 JSON 归一化
|-- metrics.py                 # 分布指标、JSD、实验评价函数
|-- atc_reference.py           # ATC 真实测试集参考指标
|-- realism_evaluation.py      # 仿真-真实数据真实性评价
|-- map_io.py                  # 后续拆分：geometry/stage/region 读写
|-- profile_sources.py         # 后续拆分：mall/llmob/atc/timb profile provider
|-- route_planner.py           # 后续拆分：baseline/LLM/检索增强路线规划
|-- journey_builder.py         # 后续拆分：route -> JuPedSim journey/stage
|-- simulation_runner.py       # 后续拆分：出生、主循环、拥堵、TTL
|-- experiment_summary.py      # 实验产物汇总
`-- demo_map_simulation.py     # 当前兼容入口，逐步变薄
```

短期内保留 `demo_map_simulation.py` 作为唯一仿真入口，避免破坏 `configs/scenarios/*.json` 和 `scripts/*.ps1`。

## 3. 数据契约

核心数据对象：

- `AgentProfile`：角色、子类型、偏好、动机、速度、默认停留时间、历史线索。
- `RouteActivity`：中间语义区域、动作、停留时间、目标点。
- `AgentRoute`：agent 的高层路线，包括活动列表、最终出口、速度和意图。
- `SavedAgentPlan`：保存到 `outputs/plans/*.json` 的可复现实验记录。
- `ExperimentScenario`：场景配置文件的规范化表示。

所有外部来源，包括 mall 内置画像、LLMob、ATC、TIMB/VIMN/Memento，都应最终转换为上述 profile/route 契约。

## 4. 参考代码落地方式

参考目录：

```text
E:\WeChat Files\wxid_aqbmlhq4ee522\FileStorage\File\2026-07\沈逸帆+毕业论文内容汇总\代码
```

可吸收能力：

- `工作1-TIMB`：个人 mobility profile、历史 routine 检索、VIMN/Memento 意图线索、JSD 评价指标。
- `工作2-Intent_simulated`：多 seed 分析、行为状态聚类、状态转移矩阵、论文图表。

不建议整包迁入。原因是它们依赖城市 POI、模型 checkpoint、PyTorch 训练流程和不同的数据结构。正确方式是做适配层：

```text
TIMB/LLMob routine -> AgentProfile.history_hint / preferences
VIMN/Memento hint  -> AgentProfile.intent_hint / memory_hint
city POI category  -> indoor semantic region category
evaluation JSD     -> experiment_summary distribution metrics
emergence states   -> post-processing over plan + trajectory artifacts
```

## 5. 实验矩阵

基础论文实验：

| 场景 | profile 来源 | 路线策略 | 目的 |
| --- | --- | --- | --- |
| `smoke_baseline` | mall | random | 快速验证工程可运行 |
| `baseline_random` | mall | random exit | 基础对照 |
| `baseline_nearest` | mall | nearest exit | 几何最短倾向对照 |
| `llm_exit_only` | mall | LLM exit | 检验 LLM 出口选择 |
| `llm_remote` | mall | LLM semantic route | 主方法 |
| `atc_llm_remote` | ATC | LLM semantic route | 数据驱动画像扩展 |

扩展实验：

- `llmob_llm_remote`：LLMob profile + LLM semantic route。
- `timb_hint_llm_remote`：TIMB/VIMN/Memento hint + LLM semantic route。
- `llm_no_wait`：移除 activity wait。
- `llm_no_routing_waypoints`：关闭 routing waypoints。
- `movement_avm_vs_cfsv3`：物理模型消融。

## 6. 指标体系

工程指标：

- plan validation error count
- TTL removal rate
- stuck reroute rate
- active population cap effect
- observed travel time

行为分布指标：

- role distribution
- final exit distribution
- semantic region visit distribution
- activity action distribution
- activity count distribution
- wait time distribution
- route directness proxy

对比指标：

- exit distribution JSD
- role distribution JSD
- region visit distribution JSD
- activity count distribution JSD
- travel time distribution JSD
- output comparison files: `experiment_summary_comparison.json` and `experiment_summary_comparison.csv`

涌现分析指标：

- agent behavior state clusters
- state transition matrix
- state dwell time
- congestion/TTL contribution by state
- role/state correlation

## 7. 公式定义

论文和实验报告中建议统一使用以下公式。工程实现可优先放在 `src/metrics.py` 和 `src/experiment_summary.py`。

### 7.1 Jensen-Shannon Divergence

用于比较两种策略下的角色分布、出口分布、语义区域访问分布、活动数量分布和旅行时长分布。

给定两个离散分布 \(P\) 和 \(Q\)，先定义：

\[
M = \frac{1}{2}(P + Q)
\]

则 Jensen-Shannon Divergence 为：

\[
JSD(P \parallel Q) =
\frac{1}{2} KL(P \parallel M) +
\frac{1}{2} KL(Q \parallel M)
\]

其中：

\[
KL(P \parallel M) =
\sum_i P_i \log \frac{P_i}{M_i}
\]

在本项目中可用于：

\[
JSD_{exit} = JSD(D^{method}_{exit} \parallel D^{baseline}_{exit})
\]

\[
JSD_{region} = JSD(D^{method}_{region} \parallel D^{baseline}_{region})
\]

### 7.2 TTL Removal Rate

用于衡量 agent 因长时间未完成路线而被移除的比例。

\[
R_{TTL} =
\frac{N_{TTL}}{N_{agents}}
\]

其中 \(N_{TTL}\) 是 TTL 移除人数，\(N_{agents}\) 是该场景总 agent 数。

### 7.3 Reroute Rate

用于衡量因拥堵或卡住触发重路由的比例。

\[
R_{reroute} =
\frac{N_{reroute}}{N_{agents}}
\]

### 7.4 Observed Travel Time

对于 agent \(i\)，若其首次和最后一次被轨迹数据库观测到的帧分别为 \(f_i^{start}\)、\(f_i^{end}\)，轨迹帧率为 \(FPS\)，则观测旅行时间为：

\[
T_i =
\frac{f_i^{end} - f_i^{start}}{FPS}
\]

平均观测旅行时间：

\[
\bar{T} =
\frac{1}{N}\sum_{i=1}^{N}T_i
\]

### 7.5 Path Length

若 agent \(i\) 的轨迹点序列为 \((x_{i,t}, y_{i,t})\)，路径长度为：

\[
L_i =
\sum_{t=1}^{T_i-1}
\sqrt{
(x_{i,t+1}-x_{i,t})^2 +
(y_{i,t+1}-y_{i,t})^2
}
\]

可进一步计算平均路径长度：

\[
\bar{L} =
\frac{1}{N}\sum_{i=1}^{N}L_i
\]

### 7.6 Mean Speed

使用路径长度和观测旅行时间定义 agent 平均速度：

\[
V_i =
\frac{L_i}{T_i}
\]

群体平均速度：

\[
\bar{V} =
\frac{1}{N}\sum_{i=1}^{N}V_i
\]

### 7.7 Activity Count

若 agent \(i\) 的中间活动序列为 \(A_i\)，活动数量为：

\[
C_i = |A_i|
\]

平均活动数量：

\[
\bar{C} =
\frac{1}{N}\sum_{i=1}^{N}C_i
\]

### 7.8 Wait Time

若 agent \(i\) 的活动集合中每个活动 \(a\) 的等待时间为 \(w_{i,a}\)，则计划等待总时长为：

\[
W_i =
\sum_{a \in A_i} w_{i,a}
\]

平均计划等待时长：

\[
\bar{W} =
\frac{1}{N}\sum_{i=1}^{N}W_i
\]

### 7.9 Exit Utilization Entropy

用于衡量出口使用是否集中。若出口分布为 \(P_{exit}\)，则出口熵为：

\[
H_{exit} =
-\sum_i P_{exit,i}\log P_{exit,i}
\]

归一化出口熵：

\[
H^{norm}_{exit} =
\frac{H_{exit}}{\log K}
\]

其中 \(K\) 是出口数量。值越高表示出口利用越均匀。

### 7.10 State Transition Matrix

若涌现分析中将 agent 行为聚类为 \(K\) 个状态，状态转移计数为：

\[
C_{ab} =
\sum_i \sum_t
\mathbb{I}(s_{i,t}=a, s_{i,t+1}=b)
\]

状态转移概率矩阵为：

\[
P_{ab} =
\frac{C_{ab}}{\sum_{b'=1}^{K} C_{ab'}}
\]

该公式对应后续 `emergence_analysis.py` 中的状态转移矩阵、Sankey 图和行为状态驻留分析。

## 8. 重构阶段

阶段 1：稳定契约和文档。

- 新增 `sim_schema.py`。
- 新增 `metrics.py`。
- 新增 `validate_scenarios.py`。
- 新增 `atc_reference.py`。
- 新增 `realism_evaluation.py`。
- 修复 prompt 文本可读性。
- 增强 `experiment_summary.py`。

阶段 2：拆分地图和路线规划。

- 抽出 `map_io.py`。
- 抽出 `route_planner.py`。
- 对 baseline/LLM route 做统一接口。

阶段 3：拆分仿真执行。

- 抽出 `journey_builder.py`。
- 抽出 `simulation_runner.py`。
- 把拥堵和 TTL 控制独立为策略函数。

阶段 4：参考代码适配。

- 新增 `profile_sources.py`，统一 mall/llmob/atc/timb。
- 新增 `timb_adapter.py`，只读取 TIMB/LLMob 产物，不依赖训练过程。
- 将 VIMN/Memento 输出作为 prompt hint，而不是放进仿真主循环。

阶段 5：论文图表和涌现分析。

- 新增 `emergence_analysis.py`。
- 新增 `plot_experiment_figures.py`。
- 生成 transition matrix、Sankey、distribution heatmap。

## 9. 工程约束

- `scripts/run_scenario.ps1` 和 `configs/scenarios/*.json` 必须长期兼容。
- 输出产物必须可追溯到 scenario、seed、git commit、地图 hash。
- 所有外部 profile 来源必须能 fallback 到 mall profile。
- LLM 输出必须经过 schema/范围校验。
- 新指标不能假设固定 JuPedSim SQLite schema，除非先做 schema 检测。
