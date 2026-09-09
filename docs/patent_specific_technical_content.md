# 专利具体技术内容整理

> 本文用于沉淀可写入专利交底书或专利说明书的技术细节。重点描述可实现步骤、数据结构、算法流程、模块接口、实施例和可替换方案。

## 1. 核心技术主线

本项目的技术主线可以凝练为：

```text
真实轨迹数据
  -> 统一坐标和语义地图
  -> 轨迹事实提取
  -> 行人画像生成
  -> 真实轨迹投影为结构化行为计划
  -> 构建 SFT 样本和 DPO 偏好样本
  -> 训练或约束大模型输出室内行为计划
  -> 转换为物理仿真可执行 journey/stage
  -> 大规模室内人群仿真
  -> 与真实轨迹分布对比评价
```

其中已经工程实现的部分包括画像、路线规划、室内语义仿真、实验矩阵、指标和报告；待实现但技术路线已确定的部分包括完整 SFT/DPO 数据集构建和模型训练侧。

## 2. 数据对象与字段

### 2.1 语义地图对象

语义地图由几何空间、入口出口和语义区域构成。

```json
{
  "geometry": "walkable area WKT",
  "stages": {
    "entrances": [
      {"name": "entrance_1", "kind": "entrance", "polygon": [[x, y], "..."]}
    ],
    "exits": [
      {"name": "exit_1", "kind": "exit", "polygon": [[x, y], "..."]}
    ]
  },
  "regions": [
    {
      "name": "region_1",
      "points_world": [[x, y], "..."],
      "description": "semantic description for planner",
      "category": "shop|corridor|restaurant|stairs|exit|entrance",
      "metadata": {}
    }
  ]
}
```

技术要点：

- 地图几何和真实轨迹均统一到米制世界坐标；
- 区域名称全局唯一；
- 出生区域具有候选出口集合；
- 语义区域用于大模型规划，几何 polygon 用于真实轨迹归属和仿真 waypoint。

### 2.2 行人画像 AgentProfile

```json
{
  "schema_version": "1.0",
  "profile_id": "profile_x",
  "role": "commuter|shopper|staff|visitor",
  "subtype": "optional subtype",
  "preferences": ["region_1", "wide corridor", "direct exit"],
  "desired_speed_mps": 1.34,
  "default_wait_seconds": 30,
  "source": "mall|atc|llmob",
  "history_hint": "historical movement clue",
  "intent_hint": "intent clue",
  "memory_hint": "memory clue",
  "metadata": {
    "source_fingerprint": "sha256:...",
    "partition": "train|tuning|evaluation"
  }
}
```

画像生成规则：

- 根据速度统计映射 `desired_speed_mps`；
- 根据区域访问频次映射 `preferences`；
- 根据停留行为映射 `default_wait_seconds`；
- 根据访问序列和时段映射 `role` 或 `subtype`；
- 画像生成记录来源、过滤条件、缓存键和 fallback 原因。

### 2.3 行为计划 AgentRoute

```json
{
  "schema_version": "1.0",
  "agent_id": "agent_001",
  "exit_idx": 2,
  "regions": ["shop_a", "corridor_b"],
  "activities": [
    {
      "region": "shop_a",
      "action": "visit|wait|shop|pass",
      "wait_seconds": 45,
      "target_position": [12.3, 4.5]
    }
  ],
  "intent": "brief natural language intent",
  "desired_speed_mps": 1.25,
  "planner": "llm_semantic|llm_exit_only|nearest|random|replay",
  "fallback_reason": ""
}
```

校验规则：

- `exit_idx` 必须存在且属于出生区域候选出口；
- `regions` 和 `activities.region` 必须属于语义地图已有区域；
- `wait_seconds` 在 0-180 秒范围；
- `desired_speed_mps` 在合理速度范围；
- 活动区域到最终出口应在路由图上可达；
- 对批量规划，返回 agent 集合必须与请求 agent 集合一致。

## 3. 真实轨迹转换为行为计划的算法

### 3.1 输入

真实轨迹点序列：

```text
track = [
  (time, person_id, x_raw, y_raw, z_raw, speed_raw, direction_raw),
  ...
]
```

配置：

- 坐标变换参数；
- 语义地图；
- 最短有效轨迹点数；
- 最短持续时间；
- 低速和停留阈值；
- 人员分区规则；
- 输出 schema 版本。

### 3.2 坐标变换

对每个轨迹点执行：

```text
x_m = origin_x + scale_x * raw_x
y_m = origin_y + scale_y * raw_y
[x_rot, y_rot] = rotate([x_m, y_m], rotation_deg)
[x_world, y_world] = [x_rot + translation_x, y_rot + translation_y]
```

可选处理：

- x/y 轴交换；
- x/y 翻转；
- 毫米到米转换；
- 异常点过滤；
- 越界率统计。

### 3.3 人员级稳定切分

示例算法：

```text
bucket = int(sha256(person_id)[0:8], 16) % 10
if bucket == 0: partition = "evaluation"
elif bucket == 1: partition = "tuning"
else: partition = "train"
```

技术效果：

- 同一人员所有轨迹窗口落入同一分区；
- 避免训练样本和评价参考之间的人员泄漏；
- 切分结果与输入顺序无关。

### 3.4 轨迹事实提取

对合格轨迹计算：

```text
duration_s = last_time - first_time
path_length_m = sum(distance(point_i, point_i+1))
mean_speed_mps = path_length_m / duration_s
slow_ratio = slow_seconds / duration_s
```

区域访问：

```text
for each point:
  region = first polygon covering point
compress consecutive identical regions
calculate dwell time per region
```

出口映射：

```text
final_point = last valid point
final_exit = nearest exit whose polygon or buffer covers final_point
if no exit matched:
  mark failure_reason = "missing_exit"
```

### 3.5 轨迹真值投影

将区域访问序列转换为活动：

```text
for each region segment:
  if dwell_time >= dwell_threshold:
    activity.action = "wait" or category-specific action
    activity.wait_seconds = clamp(dwell_time, 0, 180)
  else:
    keep as pass-through region or remove according to config
```

输出 `AgentRoute`：

```text
route.agent_id = derived agent id
route.exit_idx = mapped final exit
route.regions = intermediate semantic regions
route.activities = dwell-derived activities
route.intent = short description derived from profile and region sequence
route.desired_speed_mps = profile desired speed
route.planner = "ground_truth_projection"
```

失败样本处理：

- 不静默改写为合法计划；
- 记录 `outside_map`、`ambiguous_region`、`unreachable_transition`、`missing_exit` 等失败原因；
- 可用于数据质量报告，也可作为少量负例来源。

## 4. SFT 样本构建

### 4.1 通用样本契约

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
    "profile_version": "1.0",
    "region_map_hash": "sha256:...",
    "coordinate_transform_version": "atc_to_mall_v1",
    "metrics_version": "mobility_metrics_v2"
  },
  "input": {
    "profile": {},
    "spawn": {},
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

### 4.2 Chat 样本

```json
{
  "example_id": "sha256:...",
  "messages": [
    {
      "role": "system",
      "content": "You are an indoor pedestrian route planner. Return strict JSON..."
    },
    {
      "role": "user",
      "content": "profiles, candidate exits, semantic regions, constraints..."
    },
    {
      "role": "assistant",
      "content": "{\"plans\":[...]}"
    }
  ]
}
```

### 4.3 样本生成步骤

1. 读取训练分区轨迹窗口；
2. 生成或读取该窗口对应画像；
3. 根据出生区域生成候选出口；
4. 拼接语义地图区域名称、类别、描述和中心点；
5. 将真值投影计划序列化为严格 JSON；
6. 计算 `example_id = sha256(canonical_json(input + target + contracts))`；
7. 写入 JSONL；
8. 生成 dataset manifest，记录样本数、失败数、hash、分区和 schema 版本。

## 5. DPO 偏好样本构建

### 5.1 候选来源

对同一个 prompt 生成候选集合：

```text
candidates = [
  ground_truth_projection,
  teacher_model_candidate_1,
  teacher_model_candidate_2,
  rule_nearest_candidate,
  rule_random_candidate,
  perturbed_wrong_exit_candidate,
  perturbed_long_route_candidate,
  perturbed_profile_mismatch_candidate
]
```

### 5.2 硬校验

硬校验失败项：

- JSON 无法解析；
- 不符合 response schema；
- 缺少 agent 或 agent 重复；
- 出口不在候选出口集合；
- 区域不存在；
- 路线不可达；
- 等待时间、速度超出范围。

硬校验结果作为候选的基础质量分，同时写入失败原因。

### 5.3 软评分

示例评分：

```text
score =
  w_route * route_similarity
  + w_exit * exit_accuracy
  + w_dwell * dwell_plausibility
  + w_profile * profile_alignment
  + w_flow * flow_efficiency
  + w_format * completeness
```

各分项可定义如下：

- `route_similarity`：区域序列与真实投影序列的 LCS、编辑距离或 Jaccard；
- `exit_accuracy`：最终出口是否匹配真实轨迹出口；
- `dwell_plausibility`：停留时间是否接近真实停留或角色默认停留；
- `profile_alignment`：区域类别和活动是否符合画像偏好；
- `flow_efficiency`：路径长度、绕行程度或拥堵风险；
- `completeness`：agent 数量、字段完整性和 schema 合法性。

### 5.4 偏好对生成

```text
sort candidates by score
for candidate_i, candidate_j:
  if score_i - score_j >= min_gap:
    chosen = candidate_i
    rejected = candidate_j
    emit DPO pair
```

约束：

- chosen/rejected 尽量均为合法 JSON，避免模型只学习格式；
- 同一 prompt 的候选不得跨数据分区；
- 低分差样本进入隔离区；
- 完全非法候选只作为少量格式负例。

## 6. 人群种类模拟方法

### 6.1 角色分类

系统可基于轨迹统计和区域偏好生成角色：

- `commuter`：路径较直接、停留少、偏向入口出口或通道；
- `shopper`：区域访问多、停留时间长、偏好店铺或餐饮；
- `staff`：速度稳定、可能访问服务区或固定区域；
- `visitor`：中等速度、行为偏好弱或混合。

### 6.2 人群生成

对每个仿真场景：

```text
for each entrance:
  allocate agent count
  for each agent:
    sample profile according to profile provider
    assign spawn time using interval + jitter
    attach candidate exits according to entrance
    call planner to generate AgentRoute
```

可配置参数：

- agent 总数；
- 出生间隔；
- 出生抖动；
- 最大活跃人数；
- 画像来源；
- 路由策略；
- LLM batch size；
- 生命周期上限；
- 运动模型参数。

### 6.3 大模型规划约束

大模型输入包含：

- agent 画像；
- 出生区域；
- 候选出口；
- 语义区域列表；
- 输出 JSON schema；
- 区域和出口选择限制；
- 最大中间区域数量；
- 等待时间和速度范围。

输出必须为：

```json
{
  "plans": [
    {
      "agent_id": "agent_001",
      "role": "shopper",
      "intent": "visit a preferred shop before leaving",
      "activities": [
        {"region": "shop_a", "action": "shop", "wait_seconds": 60}
      ],
      "final_exit": "exit_2",
      "desired_speed_mps": 1.2
    }
  ]
}
```

## 7. 行为计划到物理仿真的转换

### 7.1 Journey 构建

对每个 agent route：

```text
route.activities[0] -> waiting stage or waypoint
route.activities[1] -> waiting stage or waypoint
...
route.final_exit -> exit stage
```

若启用 routing engine，系统在相邻语义点之间插入导航 waypoint，避免 agent 直线穿墙或聚集在区域中心。

### 7.2 等待控制

当 agent 到达 activity region：

```text
record wait_started
hold or slow agent for wait_seconds
release to next stage
record wait_released
```

### 7.3 拥堵和生命周期控制

系统周期性检查 agent 位移：

```text
if movement distance < stuck_threshold for N checks:
  release waiting controls or reroute to alternative exit
```

生命周期：

```text
if current_time - spawn_time > max_lifetime:
  remove agent
  mark terminal_state = ttl_removed
```

技术效果：

- 防止仿真因局部卡死无法结束；
- 记录拥堵和重路由作为评价指标，而不是隐藏异常。

## 8. 大规模实验矩阵

示例矩阵：

| 条件 | 画像来源 | 路由策略 | 作用 |
|---|---|---|---|
| B1 | mall | random | 基础规则对照 |
| B2 | mall | nearest | 几何最短对照 |
| A1 | mall | LLM exit-only | 出口选择消融 |
| M1 | mall | LLM semantic | 主方法 |
| P1 | ATC | LLM semantic | 真实轨迹画像扩展 |
| P2 | LLMob | LLM semantic | 活动数据画像扩展 |

训练实现后可扩展：

| 条件 | 模型 | 作用 |
|---|---|---|
| M0 | base prompt | 未微调大模型 |
| M1 | SFT | 监督微调贡献 |
| M2 | SFT + DPO | 偏好优化贡献 |
| A_NO_PROFILE | trained model without profile | 画像消融 |
| A_NO_DPO | SFT only | DPO 消融 |

实验矩阵技术特征：

- 配对 seed；
- 独立 run_id；
- plan fingerprint；
- scenario hash；
- 输入文件 hash；
- 模型和 prompt 版本；
- 失败重试保留历史；
- 并发执行和 LLM 并发限制。

## 9. 输出产物

每次仿真输出目录：

```text
outputs/runs/<run_id>/
  manifest.json
  resolved_scenario.json
  plans.json
  trajectory.sqlite
  events.jsonl
  llm_calls.jsonl
  metrics.json
  run.log
```

标准分析产物：

```text
agent_metrics.csv
run_metrics.csv
trajectory_points.csv
region_transitions.csv
metrics.json
statistics.csv
realism.json
experiment_report.md
```

## 10. 真实性评价

真实轨迹和仿真轨迹不逐人对齐，而采用分布一致性：

```text
P = real distribution
Q = simulated distribution
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
M = 0.5 * (P + Q)
```

比较维度：

- 旅行时间分布；
- 路径长度分布；
- 速度分布；
- 低速比例分布；
- 区域访问分布；
- 出口分布；
- 行为状态转移。

综合评分：

```text
score = 1 / (1 + weighted_distribution_distance)
```

## 11. 可写入专利的流程步骤

一种室内人群仿真方法可包括：

1. 获取真实室内行人轨迹和室内语义地图；
2. 根据坐标变换配置将真实轨迹映射至语义地图坐标系；
3. 按人员标识哈希将轨迹数据稳定划分为训练、调参和评估分区；
4. 从轨迹点序列提取持续时间、路径长度、速度、区域访问、停留和出口映射；
5. 根据轨迹事实生成行人画像；
6. 将轨迹事实投影为包含中间区域、活动、停留、最终出口和速度的结构化行为计划；
7. 根据画像、语义地图上下文和结构化行为计划生成 SFT 样本；
8. 为同一输入生成多个候选行为计划，经过硬校验和软评分构造 DPO 偏好样本；
9. 基于 SFT 样本和 DPO 样本训练路线规划模型；
10. 在仿真运行时调用路线规划模型生成多个 agent 的行为计划；
11. 将行为计划转换为物理仿真引擎的 journey、stage 或 waypoint；
12. 执行室内人群仿真，并记录轨迹、事件和运行清单；
13. 从仿真输出提取指标，并与真实轨迹参考分布进行真实性评价。

## 12. 可写入专利的系统模块

一种室内人群仿真系统可包括：

1. 真实轨迹接入模块；
2. 坐标标定模块；
3. 语义地图管理模块；
4. 人员分区和泄漏审计模块；
5. 轨迹事实提取模块；
6. 行人画像生成模块；
7. 真值行为计划投影模块；
8. SFT 样本构建模块；
9. DPO 偏好样本构建模块；
10. 候选行为计划校验与评分模块；
11. 大模型训练和模型登记模块；
12. 大模型行为规划服务模块；
13. 物理仿真转换模块；
14. 室内人群仿真执行模块；
15. 指标分析和真实性评价模块；
16. 可追溯运行清单生成模块。

## 13. 关键可保护细节

建议在后续权利要求中重点体现：

- 真实轨迹点序列向语义活动计划的确定性投影；
- 样本契约同时绑定 prompt、response schema、地图 hash、坐标变换版本、人员分区和指标版本；
- 同一轨迹窗口生成 SFT 与 DPO 样本；
- DPO 候选先硬校验再软评分；
- 人群画像与真实轨迹统计、区域访问和停留行为关联；
- 大模型输出经 schema、候选出口和可达性校验后才进入物理仿真；
- fallback 不隐藏，而作为指标进入报告；
- 仿真结果与真实轨迹按分布对比，而非逐人轨迹误差。

## 14. 当前项目与技术内容的对应关系

| 技术内容 | 当前工程状态 |
|---|---|
| 语义地图、入口出口、区域 | 已实现 |
| AgentProfile/AgentRoute 数据契约 | 已实现 |
| mall/ATC/LLMob 画像接入 | 已实现 |
| LLM 路径规划与校验 | 已实现 |
| 物理仿真、等待、拥堵、TTL | 已实现 |
| 实验矩阵、manifest、报告 | 已实现 |
| 轨迹指标和真实性评价 | 已实现 |
| SFT 样本契约 | docs 中已有完整路线，待编码 |
| DPO 样本构建和评分器 | docs 中已有完整路线，待编码 |
| QLoRA-SFT/DPO 训练 | docs 中已有完整路线，待编码 |
| 训练模型服务化接入 | docs 中已有完整路线，待编码 |

## 15. 后续补充材料建议

为提高专利交底质量，建议继续补充：

1. 一张总体架构图；
2. 一张真实轨迹到 SFT/DPO 样本流程图；
3. 一张行为计划进入 JuPedSim 的流程图；
4. 一个真实轨迹转换样本的完整 JSON 示例；
5. 一个 DPO chosen/rejected 偏好对示例；
6. 一个仿真 run manifest 示例；
7. 一个仿真报告中的真实性评价示例。

