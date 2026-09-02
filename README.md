# COVID-19 冬季寒冷假说分析 — 代码与产出物说明

用[约翰霍普金斯大学 CSSE 新冠疫情](https://github.com/CSSEGISandData/COVID-19)时间序列（2020-01-22 ~ 2023-03-09）叠加 ERA5 实测气温，检验"冬天更冷的地区，COVID-19 死亡是否更糟"。

完整论证与所有数字见 **`冬天真的更致命吗.md`**（约 2.4 万字，6 张图）。本文件只说明代码怎么跑、产出物是什么。

**三条核心结论**（ reproduced by `step19_hemisphere.py` + `step15_figures.py`）：

| 结论                    | 数字                                                         | 来源文件                                                               |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 冬天确实更致命，机制是传播增强而非毒力增强 | 美国冬季死亡 +149%~+152%、确诊 +191%，冬季病死率反而低 8%~14%                | `winter_penalty_us_lag21.csv`                                      |
| 越冷的地方，冬季亏吃得越大（剂量—反应）  | 北半球 r = -0.40（n=163, p<10⁻⁷），美国 r = -0.44（n=102, p=4×10⁻⁶） | `hemi_dose_global.csv`、`hemi_us.csv`、`winter_penalty_us_lag21.csv` |
| 两个半球方向一致，幅度与冬季温差成比例   | 北半球（温差 14.3 ℃）死亡超额 +208%，南半球（温差 4.7 ℃）+51%                 | `hemi_global.csv`、`figures/fig6_hemisphere.png`                    |

---

## 一、运行环境

| 项   | 值                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------- |
| 解释器 | `C:\Users\37516\AppData\Local\Programs\Python\Python313\python.exe`（系统 Python 3.13）                                 |
| 依赖  | pandas 2.2.1、numpy、scipy、matplotlib。**不需要 statsmodels** — 双向固定效应泊松回归在 `step4_regression.py` 内自行实现（IRLS + 地区聚类稳健标准误） |
| 网络  | 仅 `step2_temp.py` 与 `step9_coords.py` 需要联网（抓 Open-Meteo）。抓取结果写入 `temp_cache*.json`，重跑命中缓存，不会重复请求                    |
| 字体  | 绘图用 `Microsoft YaHei / SimHei / DengXian`，Windows 下开箱可用                                                             |

## 二、输入数据

| 数据                                                                                                                | 位置 / 来源                                                                | 用途                |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------- |
| [约翰霍普金斯大学 CSSE 新冠疫情](https://github.com/CSSEGISandData/COVID-19) 时间序列（5 份：全球+美国的 confirmed / deaths，全球 recovered） | 路径常量在 **`cw_core.py:29` 的 `DATA_DIR`**，默认为./JHU_CSSE                   | 全部疫情指标            |
| 逐日气温（ERA5 再分析，`temperature_2m_mean`）                                                                              | Open-Meteo Archive API `https://archive-api.open-meteo.com/v1/archive` | 把"寒冷"从纬度代理升级为实测气温 |

原始 JHU 序列有两个必须处理的坑，都在 `cw_core.py` 里解决：

1. **回调（负增量）**：72/115 个国家、50/52 个州出现过累计值向下修订。用**回溯剥离法**修复，保证修复后单调不减、终值严格守恒。
2. **大国坐标错误**：JHU 给的是地理中心（俄罗斯落在中西伯利亚），而气温需要人口/疫情中心。`step9_coords.py` 用一张 `OVERRIDE` 表覆盖 16 个大国的坐标（俄、加、美、巴西、印度、日本等）后重新抓气温。

## 三、复现顺序

全部脚本按依赖顺序执行即可从零重建（除论文正文外）：

```bash
git clone https://github.com/zyq5945/covid_winter.git
cd covid_winter
PY="C:/Users/37516/AppData/Local/Programs/Python/Python313/python.exe"

# pip install pandas numpy scipy matplotlib

# 基础层
$PY step1_qc.py          # 质检 + 回调修复不变量校验
$PY step2_temp.py        # 需联网：抓取 v1 气温（原坐标）
$PY step9_coords.py      # 需联网：抓取 v2 气温（修正坐标，后续全部脚本用 v2）

# 回归支线（论文 5—6 节的稳健性讨论）
$PY step3_analysis.py
$PY step4_regression.py
$PY step5_hetero.py
$PY step6_robust.py
$PY step7_clean.py
$PY step8_phase.py
$PY step10_refit.py
$PY step11_reconcile.py
$PY step12_decompose.py
$PY step13_burden.py

# 决定性检验支线（论文主证据）
$PY step14_decisive.py   # 美国 + 全球旧口径
$PY step19_hemisphere.py # 按半球定季节，取代全球旧口径
$PY step15_figures.py    # 图 1—5，必须在 step14 与 step19 之后
$PY step20_hemi_fig.py   # 图 6
$PY step22_verify_paper.py  # 可选：把论文 4.1—4.4 节关键数字与现行 CSV 逐项核对
```

`step21_unbold.py` 是一次性的排版工具（精简论文加粗），改稿后如需重跑再执行；它会自动备份原文件为 `*.bak_bold`。

## 四、代码地图

### 基础层

| 脚本            | 职责                                                           | 主要产出                                 |
| ------------- | ------------------------------------------------------------ | ------------------------------------ |
| `cw_core.py`  | **公共模块**。加载 JHU 序列 → 地区聚合 → 回调修复（回溯剥离） → 7 日平滑 → 波次分层 → 季节标注 | 无（被 step1/2/3/4/9 引用）                |
| `step1_qc.py` | 量化回调规模，验证"零负增量 + 终值守恒"两道不变量                                  | `qc_global.csv`（115）、`qc_us.csv`（52） |

### 气温层

| 脚本                | 职责                                                         | 主要产出                                            |
| ----------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| `step2_temp.py`   | 按**病例加权经纬度中心**抓 v1 气温                                      | `daily_temperature.csv`、`temp_cache.json`       |
| `step9_coords.py` | 修正 16 个大国坐标（俄/加/巴西/印度/日本等）后重抓，气温窗口 2019-12-01 ~ 2023-04-30 | `daily_temperature_v2.csv`、`temp_cache_v2.json` |

### 回归支线（支撑论文 5—6 节：CFR 为何不稳健）

| 脚本                    | 职责                                                            | 主要产出                                              |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| `step3_analysis.py`   | 时序内对照（同一流行年内寒季 vs 暖季）+ 配对检验，滞后对齐 CFR                          | `pairs_{us,global}.csv`、`monthly_{us,global}.csv` |
| `step4_regression.py` | **双向固定效应泊松回归**（地区 FE + 日历月 FE），自建 `poisson_fe()`              | `fe_regressions.csv`、`panel_{scope}_{样本}.csv`     |
| `step5_hetero.py`     | 异质性：季节温差越大的地区效应是否越负                                           | `heterogeneity.csv`                               |
| `step6_robust.py`     | 温度分箱（形状诊断）+ 留一地区法 + 多 lag 扫描                                  | `tbins_{us,global}.csv`、`loo_{us,global}.csv`     |
| `step7_clean.py`      | 半球×日历月 FE；2022 年（自测漏报）剔除                                      | `spec_robustness.csv`                             |
| `step8_phase.py`      | **关键证伪**：效应是否只是"疫情阶段"混杂（加 log 累计确诊 / 地区趋势）                    | `phase_confound.csv`                              |
| `step10_refit.py`     | 用 v2 气温重估，坐标修正前后对比；**同时提供共享函数 `prep_tp()` 给 step11—15、19 调用** | `coord_fix.csv`、`global_bands_v2.csv`             |
| `step11_reconcile.py` | 参数模型 vs 非参数事实的口径对账                                            | `nonparam_{scope}_{all,post}.csv`                 |
| `step12_decompose.py` | 把双向 FE 的系数拆成横截面维度与时序维度                                        | 控制台输出                                             |
| `step13_burden.py`    | 改用不受检测口径污染的结局：死亡负担与传播强度                                       | `burden_transmission.csv`（需重跑，当前不在磁盘）             |

### 决定性检验支线（论文主证据）

| 脚本                     | 职责                                                                                                    | 主要产出                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `step14_decisive.py`   | **核心设计**：每个冬天夹在前后两个夏天中间，用两夏平均做对照，一阶抵消疫苗/毒株/检测等平滑趋势。美国样本无半球问题，结果现行有效                                   | `winter_penalty_{us,global}_lag{14,21,28}.csv`                            |
| `step19_hemisphere.py` | **修正季节反转 bug**。原实现按日历月硬编码（11—3 月=冬），导致南半球 17 国（占全球样本死亡 22.1%）季节全反。改为按半球定义，并做单侧对照分解（仅前夏 / 仅后夏）识别奥密克戎污染 | `hemi_global.csv`、`hemi_us.csv`、`hemi_dose_global.csv`、`hemi_dose_us.csv` |
| `step16_dose.py`       | 剂量—反应回归（旧口径，基于 step14 的全球表），已被 step19 的 `hemi_dose_*` 取代                                              | `dose_response.csv` 等（已废弃）                                                |
| `step15_figures.py`    | 图 1—5 + 汇总表。读 `winter_penalty_us_lag21.csv`（美国）与 `hemi_global.csv`（全球），**必须在 step14、step19 之后跑**      | `figures/fig1`—`fig5`、`final_penalty_table.csv`                           |
| `step20_hemi_fig.py`   | 图 6：南北半球对照与宏观剂量—反应                                                                                    | `figures/fig6_hemisphere.png`                                             |

## 五、产出物

### 5.1 论文直接引用（交付物）

| 文件                            | 行数       | 内容                                           | 论文位置                  |
| ----------------------------- | -------- | -------------------------------------------- | --------------------- |
| `冬天真的更致命吗.md`                 | —        | 最终论文                                         | 全文                    |
| `figures/fig1_us_monthly.png` | —        | 美国合计逐月走势（确诊 / 归因死亡 / 滞后对齐 CFR + 气温）          | 图 3                   |
| `figures/fig2_penalty.png`    | —        | 决定性检验总览（1×3：美国 / 北半球 / 南半球）                  | 图 2                   |
| `figures/fig3_scatter.png`    | —        | 剂量—反应散点（美国 / 北半球）                            | 图 5                   |
| `figures/fig4_monthshare.png` | —        | 死亡与确诊在一年中的月份分布                               | 图 4                   |
| `figures/fig5_dataqc.png`     | —        | 回调最严重的 12 国 + 12 州                           | 图 1                   |
| `figures/fig6_hemisphere.png` | —        | 南北半球对照 + 宏观剂量—反应                             | 图 6                   |
| `hemi_global.csv`             | 178      | 全球 93 国 × 各冬季窗口的超额与气温（现行口径）                  | 3.4、4.1、4.2、4.4 节     |
| `hemi_us.csv`                 | 102      | 美国 52 地区 × 2 个冬天（含每百万人口径）                    | 同上                    |
| `hemi_dose_global.csv`        | 3        | 分半球分窗口的 Pearson / Spearman / r² / p（死亡绝对值口径） | 4.3 节（北半球）、4.4 节（南半球） |
| `hemi_dose_us.csv`            | 2        | 同上（美国，**绝对死亡口径**，论文 4.3 未直接引用；美国正文用的是每百万人口径） | step19 产出             |
| `winter_penalty_us_lag21.csv` | 102      | 美国决定性检验明细（现行有效）                              | 4.1—4.3 节             |
| `final_penalty_table.csv`     | 15       | 三组样本 × 三个指标 × 各冬天的效应幅度 / 一致占比 / p 值汇总表       | 4.1 节表                |
| `qc_global.csv` / `qc_us.csv` | 115 / 52 | 每个地区的回调量、占比、最终累计值                            | 图 1 数据源               |
| `coord_fix.csv`               | 12       | 坐标修正前后的系数对比（证明测量误差会把效应拉向 0）                  | 3.3 节                 |

**`hemi_*.csv` 列命名规则**（读表前必看）：

| 列                                                  | 含义                                         |
| -------------------------------------------------- | ------------------------------------------ |
| `hemi`                                             | 北半球 / 南半球                                  |
| `winter`                                           | 冬季窗口编号，如 `W1 2020-21冬`                     |
| `m_w` / `m_s1` / `m_s2`                            | 冬窗、前夏、后夏的有效月数（用于判断是否完整）                    |
| `temp_w` / `temp_s`                                | 冬季、夏季（两夏平均）日均温，℃                           |
| `penalty_cases` / `penalty_deaths` / `penalty_cfr` | 对数尺度超额 = log(冬季值 ÷ 前后两夏均值)。**正值 = 冬季更糟**   |
| `penalty_*_pre` / `penalty_*_post`                 | 单侧对照：仅用前一个夏天 / 仅用后一个夏天。两侧差异大 ⇒ 对照窗口被毒株更替污染 |
| `penalty_deaths_pm`                                | 每百万人死亡口径的超额（美国样本才有）                        |

### 5.2 支撑性中间产物（复核用，不进入最终结论）

`daily_temperature_v2.csv`（现行气温表，6.6 MB）、`monthly_{us,global}.csv`、`pairs_{us,global}.csv`、`panel_*.csv`、`fe_regressions.csv`、`heterogeneity.csv`、`spec_robustness.csv`、`phase_confound.csv`、`loo_{us,global}.csv`、`nonparam_*.csv`、`global_bands_v2.csv`、`winter_penalty_us_lag14.csv` / `lag28.csv`（滞后稳健性）。

## 六、口径备忘（改代码前必读）

1. **季节必须按半球定义**：北半球冬 11—3 月、夏 5—9 月；南半球冬 5—9 月、夏 11—3 月；4 月与 10 月为过渡月，主分析剔除。按日历月一刀切会让南半球全部算反。
2. **滞后对齐 21 天**为默认，稳健性扫描 14 / 28 天。死亡整体后移后再按月汇总，保证分子分母时间匹配。
3. **水平类指标（确诊数、死亡数）按月均归一**，否则长短不一的窗口不可比；病死率本身是比值，不归一。
4. **对照 = 前后两个夏天的平均值**，用于一阶抵消平滑趋势。改动这里会直接推翻主结论。
5. **锚定死亡负担，不要锚定病死率（CFR）**。CFR 分母是确诊数，受检测口径影响极大，在本数据上不存在稳健证据（拆解后横截面与时序两个维度各自为零）。
6. **气温一律用 v2**（`step9_coords.py` 产出）。v1 仅用于 `step10_refit.py` 里的修正前后对比。
7. **南半球只有 1 个可用冬季窗口**（2021 冬），且其后的夏天正是奥密克戎期。任何南半球结论都要配 `_pre` / `_post` 单侧分解一起看。

## 七、已废弃，勿引用

| 文件                                                                          | 废弃原因                                                         |
| --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `winter_penalty_global_lag{14,21,28}.csv`                                   | 按日历月定季节，南半球 17 国季节算反。已被 `hemi_global.csv` 取代（美国同名文件不受影响，仍现行） |
| `dose_response.csv`、`dose_response_cfr.csv`、`dose_global.csv`、`dose_us.csv` | 基于上面那份全球旧表算的剂量—反应。已被 `hemi_dose_*.csv` 取代                    |
| `daily_temperature.csv`、`temp_cache.json`                                   | v1 原坐标气温。仅供 step10 做修正前后对比                                   |
| `step17_report.py`、`report.html`                                            | 早期 HTML 报告，已被 Markdown 论文取代                                  |
| `step18_paper_html.py`、`冬天真的更致命吗.html`、`冬天真的更致命吗1.html`                     | HTML 版论文不再维护，只维护 `.md`                                       |
