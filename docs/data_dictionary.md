# 数据字典

## 1. 版本规则

新增的结构化产物使用 `schema_version`。批次 A 的版本为 `1.0`。读取器可以归一化缺少版本的历史数据，但正式实验写出的新产物必须显式携带版本。

对应 JSON Schema 位于 `configs/schemas`。

## 2. AgentProfile

| 字段 | 类型 | 单位/范围 | 含义 |
| --- | --- | --- | --- |
| `profile_id` | string | 非空 | 画像唯一标识 |
| `role` | enum | commuter/shopper/staff/visitor | 主要角色 |
| `subtype` | string/null | - | 细分类型 |
| `preferences` | string[] | - | 区域或行为偏好 |
| `desired_speed_mps` | number | 0.8-1.8 m/s | 期望速度 |
| `default_wait_seconds` | integer | 0-180 s | 默认活动等待时间 |
| `source` | string | mall/atc/llmob 等 | 画像来源 |
| `history_hint` | string | - | 历史活动线索 |
| `intent_hint` | string | - | 意图线索 |
| `memory_hint` | string | - | 记忆线索 |
| `metadata` | object | - | 来源专用附加字段 |

## 3. AgentRoute

| 字段 | 类型 | 单位/范围 | 含义 |
| --- | --- | --- | --- |
| `agent_id` | string | 非空 | agent 标识 |
| `exit_idx` | integer | >= 0 | 最终出口索引 |
| `regions` | string[] | 地图已有区域 | 中间语义区域序列 |
| `activities` | RouteActivity[] | - | 中间活动 |
| `intent` | string | - | 可解释自然语言意图 |
| `desired_speed_mps` | number | 0.8-1.8 m/s | 路线采用的期望速度 |
| `planner` | string | - | random/nearest/llm/replay 等规划器 |
| `fallback_reason` | string | 空表示未 fallback | 降级原因 |

## 4. RouteActivity

| 字段 | 类型 | 单位/范围 | 含义 |
| --- | --- | --- | --- |
| `region` | string | 地图已有区域 | 活动目标语义区域 |
| `action` | string | - | visit/wait/shop 等动作 |
| `wait_seconds` | integer | 0-180 s | 计划停留时间 |
| `target_position` | number[2]/null | m | 实际采样目标点 |

## 5. 地图对象

### SemanticRegion

- `name`：区域唯一名称；
- `points_world`：世界坐标 polygon，单位为米；
- `description`：供规划器使用的语义描述；
- `category`：标准化室内区域类别；
- `metadata`：像素坐标、颜色等辅助信息。

### StageDefinition

- `name`：stage 名称；
- `kind`：`entrance`、`exit`、`waypoint` 或 `waiting_set`；
- `polygon`：世界坐标 polygon，单位为米；
- `candidate_exit_indices`：出生 stage 允许的出口索引。

## 6. ExperimentScenario

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | string | 场景格式版本 |
| `name` | string | 必须与文件名一致 |
| `description` | string | 实验目的与外部前提 |
| `enabled` | boolean | 默认批量运行是否包含 |
| `args` | array | 传递给仿真入口的参数 |
| `env` | object | 该场景所需环境变量覆盖 |

参数仍以数组保存是为了兼容现有入口。后续 matrix 层会将实验因子解析为完整场景快照。

## 7. RunManifest

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `run_id` | string | 单次独立运行标识 |
| `scenario_name` | string | 场景名称 |
| `seed` | integer | 仿真基础 seed |
| `status` | enum | planned/running/completed/failed/interrupted |
| `started_at` | datetime/null | ISO 8601 开始时间 |
| `finished_at` | datetime/null | ISO 8601 结束时间 |
| `returncode` | integer/null | 进程退出码 |
| `scenario_snapshot` | object | 解析后的配置快照 |
| `reproducibility` | object | commit、环境、依赖和文件 hash |
| `artifacts` | object | 轨迹、计划、画像、日志等路径 |
| `failure` | object/null | 失败阶段、异常类型和消息 |

## 8. RunMetrics

计数型字段均以 agent 为单位：

- `spawned_agents`：成功加入仿真的人数；
- `completed_agents`：正常完成路线的人数；
- `ttl_removed_agents`：因生命周期上限移除的人数；
- `failed_agents`：因异常无法继续的人数；
- `unfinished_agents`：仿真停止时仍未进入终态的人数。

所有比例必须明确分母。例如完成率定义为：

```text
completed_agents / spawned_agents
```

`metrics` 保存标量，`distributions` 保存分箱或分类计数。缺失指标使用 `null`，不得用 `0` 冒充未知值。

## 9. 时间与空间单位

- 世界坐标：米；
- 速度：米/秒；
- 仿真时间、旅行时间、等待时间：秒；
- 帧号：无量纲整数；
- ATC 原始位置和速度可能以毫米表示，进入 processed 层时必须转换为米和米/秒；
- 像素坐标仅用于地图编辑，不得直接进入物理指标计算。

## 10. Provider 与 Planner 追溯字段

画像缓存的 `preprocessing` 字段记录：

- 请求来源与实际来源；
- 来源文件指纹和缓存键；
- seed、数据集、人数、行数和最小轨迹点过滤参数；
- 处理前后数量、拒绝数、字段缺失率、异常率、角色/来源分布、缓存命中和 fallback 原因。

计划文件的 `metadata.planning` 字段记录：

- planner policy；
- 计划人数、fallback 人数和校验错误数；
- LLM 失败 batch 数；
- prompt 版本、模型、缓存目录和调用日志位置；
- replay 来源计划及其原始 route policy。

LLM 调用日志使用 JSONL，每行代表一个 batch 调用或缓存命中。`request_key`
由模型、采样参数和完整 messages 共同计算，`prompt_hash` 只标识 prompt 内容。

## 11. 标准分析产物

- `agent_metrics.csv`：每行一个 `run_id + agent_id`，记录终态、旅行时间、路径长度、速度、直接度、慢速/停留/等待、密度、重路由及额外路径代价；
- `run_metrics.csv`：每行一个独立 run，比例分母以 `metrics.json.metric_definitions` 为准；
- `trajectory_points.csv`：标准米制轨迹点，仅用于可重建绘图和行为状态窗口；
- `region_transitions.csv`：逐运行区域转移计数；
- `metrics.json`：保存标量、分类分布、分组统计、SQLite schema 探测结果和完整指标定义。

`reroute_extra_path_m` 定义为重路由事件后观察到的实际剩余路径长度减去事件后首个轨迹点到最终轨迹点的直线距离，下限为 0；没有足够事件后轨迹时为 `null`。

ATC 人员按 `sha256(person_id)` 的前 8 个十六进制字符对 10 取模进行稳定分区：桶 0 为 `evaluation`，桶 1 为 `tuning`，其余为 `train`。P1 画像默认只读 `train`，ATC 真实性参考只读 `evaluation`；`all` 仅允许用于工程诊断，不得用于正式评价。
