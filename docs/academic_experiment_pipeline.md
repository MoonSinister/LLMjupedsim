# 论文实验流水线

本文档给出当前工程的学术实验组织方式。目标是把实验从“运行一次仿真”转为“真实数据基准、对照组、消融组、真实性评价、可复现记录”的完整流程。

## 1. 数据分层

### 1.1 地图数据

必需：

```text
data/map/geometry.wkt
data/map/stages.json
data/map/localization_grid_regions.json
```

这些文件定义仿真空间、入口出口和语义区域。

### 1.2 真实测试集

ATC 数据作为真实性评价基准：

```text
ATC raw/processed CSV
ATC region JSON
```

建议通过环境变量指定：

```powershell
$env:ATC_RAW_PATH='D:\path\to\atc-tracking'
$env:ATC_REGIONS='D:\path\to\localization_grid_regions.json'
```

### 1.3 仿真输出

```text
outputs/trajectories/*.sqlite
outputs/plans/*.json
outputs/profiles/*.json
outputs/runs/*.json
outputs/summaries/*.json
outputs/summaries/*.csv
```

## 2. 实验步骤

### Step 1：校验实验配置

```powershell
D:\anaconda\envs\python3.12\python.exe src\validate_scenarios.py
```

### Step 1.5：检查本地 ATC 单日数据

当前仓库已有一天 ATC 数据：

```text
data/atc-20121114/atc-20121114.csv
```

该文件为无表头原始 ATC 格式：

```text
time, person_id, x_mm, y_mm, z_mm, speed_mm_s, move_rad, heading_rad
```

可先用小样本确认解析和区域映射：

```powershell
D:\anaconda\envs\python3.12\python.exe src\atc_reference.py `
  --atc-raw-path data\atc-20121114\atc-20121114.csv `
  --atc-regions data\map\localization_grid_regions.json `
  --max-rows 200000 `
  --max-persons 50 `
  --min-points 100 `
  --output outputs\summaries\atc_20121114_reference_sample.json
```

如果样本输出中 `region_visits` 非空，说明真实轨迹点能映射到当前语义区域，可进入正式实验。

### Step 2：生成 ATC 真实参考集

```powershell
D:\anaconda\envs\python3.12\python.exe src\atc_reference.py `
  --atc-raw-path "$env:ATC_RAW_PATH" `
  --atc-regions "$env:ATC_REGIONS" `
  --max-persons 200 `
  --min-points 300 `
  --output outputs\summaries\atc_reference_metrics.json
```

对于仓库内的 2012-11-14 单日数据，可直接运行：

```powershell
.\scripts\build_atc_20121114_reference.ps1
```

默认输出：

```text
outputs/summaries/atc_20121114_reference_metrics.json
```

输出：

```text
outputs/summaries/atc_reference_metrics.json
```

该文件包含真实轨迹的：

- 持续时间分布
- 路径长度分布
- 平均速度分布
- 低速比例分布
- 区域访问分布

### Step 3：运行对照组和主方法

```powershell
.\scripts\run_scenario.ps1 --repeat 5 --seed-start 2026 baseline_random baseline_nearest llm_exit_only llm_remote
```

如果需要数据驱动画像：

```powershell
.\scripts\run_scenario.ps1 --include-disabled atc_llm_remote
.\scripts\run_scenario.ps1 --include-disabled llmob_llm_remote
```

### Step 4：汇总实验产物

```powershell
D:\anaconda\envs\python3.12\python.exe src\experiment_summary.py
```

输出：

```text
outputs/summaries/experiment_summary.json
outputs/summaries/experiment_summary.csv
outputs/summaries/experiment_summary_aggregate.json
outputs/summaries/experiment_summary_aggregate.csv
outputs/summaries/experiment_summary_comparison.json
outputs/summaries/experiment_summary_comparison.csv
```

### Step 5：真实性评价

```powershell
D:\anaconda\envs\python3.12\python.exe src\realism_evaluation.py `
  --reference outputs\summaries\atc_reference_metrics.json `
  --sim outputs\trajectories\baseline_random.sqlite outputs\trajectories\demo_map_llm_remote.sqlite
```

输出：

```text
outputs/summaries/realism_evaluation.json
outputs/summaries/realism_evaluation.csv
```

## 3. 论文实验矩阵

| 类型 | 场景 | 说明 |
| --- | --- | --- |
| 工程验证 | `smoke_baseline` | 快速确认地图、入口、出口和仿真主循环可运行 |
| 基础对照 | `baseline_random` | 随机出口策略 |
| 几何对照 | `baseline_nearest` | 最近出口策略 |
| 消融实验 | `llm_exit_only` | LLM 只选择出口，不规划中间语义活动 |
| 主方法 | `llm_remote` | LLM 规划中间语义活动和最终出口 |
| 数据画像扩展 | `atc_llm_remote` | ATC profile + LLM 路线 |
| 数据画像扩展 | `llmob_llm_remote` | LLMob profile + LLM 路线 |

## 4. 真实性指标

令真实 ATC 分布为 \(P\)，仿真分布为 \(Q\)。各项真实性差异使用：

\[
JSD(P \parallel Q)
\]

综合差异：

\[
D_{realism} =
w_T JSD_T +
w_L JSD_L +
w_V JSD_V +
w_R JSD_R +
w_S JSD_S
\]

其中：

- \(JSD_T\)：持续时间分布差异
- \(JSD_L\)：路径长度分布差异
- \(JSD_V\)：速度分布差异
- \(JSD_R\)：区域访问分布差异
- \(JSD_S\)：低速比例分布差异

真实性得分：

\[
Score_{realism} =
\frac{1}{1 + D_{realism}}
\]

分数越高，表示仿真结果越接近 ATC 真实行为分布。

## 5. 学术写作口径

建议在论文中说明：

```text
由于仿真场景和真实 ATC 数据不具备逐个行人的初始条件对应关系，
本文不采用点对点轨迹误差作为真实性评价，而采用分布一致性指标。
通过比较速度、路径长度、旅行时间、区域访问和低速停留等行为分布，
评估不同路线规划策略是否生成更接近真实人群的群体行为。
```
