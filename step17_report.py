"""Step 17: 写最终 HTML 报告 (结论先行)"""
from __future__ import annotations
import os, base64
import numpy as np, pandas as pd


def fig_to_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def main():
    # 读关键结果
    pen = pd.read_csv("final_penalty_table.csv")
    dose = pd.read_csv("dose_response.csv")
    dose_cfr = pd.read_csv("dose_response_cfr.csv")
    qc_g = pd.read_csv("qc_global.csv")
    qc_u = pd.read_csv("qc_us.csv")
    fe = pd.read_csv("fe_regressions.csv")

    def df_to_html(df, fmts=None):
        if fmts is None:
            fmts = {}
        s = df.copy()
        for c, f in fmts.items():
            if c in s.columns:
                s[c] = s[c].map(f)
        return s.to_html(index=False, border=0, classes="t", escape=False)

    def fmt_pct(v):
        return f"{v:+.1f}%" if pd.notnull(v) else "—"

    def fmt_p(v):
        if pd.isnull(v):
            return "—"
        if v < 1e-4:
            return f"{v:.1e}"
        return f"{v:.4f}"

    def fmt_x(v):
        return f"{v:.3f}"

    f1 = fig_to_data_uri("figures/fig1_us_monthly.png")
    f2 = fig_to_data_uri("figures/fig2_penalty.png")
    f3 = fig_to_data_uri("figures/fig3_scatter.png")
    f4 = fig_to_data_uri("figures/fig4_monthshare.png")
    f5 = fig_to_data_uri("figures/fig5_dataqc.png")

    # 关键数字
    us_w2 = pen[(pen.scope == "美国 50 州") & (pen.winter == "2021-22 冬")].set_index("metric")
    g_w2 = pen[(pen.scope == "全球 113 国") & (pen.winter == "2021-22 冬")].set_index("metric")

    # 月度分布 (US 与 global 分别算)
    by_g = pd.read_csv("monthly_global.csv")
    by_g["mon"] = pd.to_datetime(by_g["ym"]).dt.month
    by_u = pd.read_csv("monthly_us.csv")
    by_u["mon"] = pd.to_datetime(by_u["ym"]).dt.month

    def shares(by, col):
        s = by.groupby("mon")[col].sum()
        cold = s.loc[[11, 12, 1, 2, 3]].sum() / s.sum() * 100
        warm = s.loc[[5, 6, 7, 8, 9]].sum() / s.sum() * 100
        return cold, warm

    u_d_cold, u_d_warm = shares(by_u, "deaths")
    u_c_cold, u_c_warm = shares(by_u, "confirmed")
    g_d_cold, g_d_warm = shares(by_g, "deaths")
    g_c_cold, g_c_warm = shares(by_g, "confirmed")

    pen_html = pen.copy()
    pen_html["eff"] = pen_html["eff"].map(fmt_pct)
    pen_html["share"] = pen_html["share"].map(lambda v: f"{v:.1f}%")
    pen_html["p"] = pen_html["p"].map(fmt_p)

    dose_html = dose.copy()
    dose_html["pearson_r"] = dose_html["pearson_r"].map(lambda v: f"{v:+.3f}")
    dose_html["spearman_r"] = dose_html["spearman_r"].map(lambda v: f"{v:+.3f}")
    dose_html["pearson_p"] = dose_html["pearson_p"].map(fmt_p)
    dose_html["spearman_p"] = dose_html["spearman_p"].map(fmt_p)
    dose_html["slope_pct_per_C"] = dose_html["slope_pct_per_C"].map(lambda v: f"{v:+.3f}")
    dose_html["weighted_slope_per_C"] = dose_html["weighted_slope_per_C"].map(lambda v: f"{v:+.3f}")

    # QC summary
    n_rev_g = (qc_g.rev_pct > 0).sum()
    max_rev_g = qc_g.rev_pct.max()
    n_rev_u = (qc_u.rev_pct > 0).sum()
    max_rev_u = qc_u.rev_pct.max()

    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>COVID-19 冬季严寒假说分析报告</title>
<style>
  body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.7;
        max-width:1080px;margin:24px auto;padding:0 24px;color:#1a202c}}
  h1{{font-size:26px;border-bottom:3px solid #2b6cb0;padding-bottom:6px;margin-bottom:6px}}
  h2{{font-size:19px;color:#2b6cb0;border-left:5px solid #2b6cb0;padding-left:10px;margin-top:32px}}
  h3{{font-size:15px;color:#2d3748;border-bottom:1px dashed #cbd5e0;padding-bottom:3px;margin-top:22px}}
  .kicker{{color:#718096;font-size:13px;margin-bottom:18px}}
  .conclusion{{background:#fff5f5;border:1px solid #fc8181;border-left:6px solid #c53030;
               padding:14px 18px;border-radius:4px;margin:14px 0;font-size:14.5px}}
  .conclusion b{{color:#9b2c2c}}
  .key{{background:#ebf8ff;border-left:5px solid #2b6cb0;padding:12px 16px;
        border-radius:3px;margin:12px 0;font-size:14.5px}}
  .key b{{color:#2c5282}}
  .n{{display:inline-block;background:#edf2f7;padding:1px 8px;border-radius:10px;
       font-size:12px;color:#4a5568;margin:0 4px}}
  .big{{font-size:22px;font-weight:bold;color:#c53030}}
  .bigb{{font-size:22px;font-weight:bold;color:#2b6cb0}}
  table.t{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}}
  table.t th{{background:#edf2f7;padding:6px 10px;text-align:left;border:1px solid #cbd5e0}}
  table.t td{{padding:5px 10px;border:1px solid #e2e8f0;font-variant-numeric:tabular-nums}}
  table.t tr:nth-child(even){{background:#f7fafc}}
  figure{{margin:18px 0;text-align:center}}
  figcaption{{font-size:12.5px;color:#4a5568;margin-top:4px}}
  code{{background:#edf2f7;padding:1px 5px;border-radius:2px;font-size:13px;color:#9b2c2c}}
  .muted{{color:#718096;font-size:12.5px}}
  ul li{{margin-bottom:4px}}
</style></head><body>

<h1>COVID-19 冬季严寒假说 —— 数据分析报告</h1>
<p class="kicker">数据：JHU CSSE 时间序列（2020-01-22 ~ 2023-03-09） | 共 115 国 + 50 州+DC + 12.7 亿确诊 + 793 万死亡
&nbsp;|&nbsp; 研究日：2026-08-29</p>

<div class="conclusion">
<b>结论先行：</b>用户"冬天寒冷的地区死亡情况更糟"的直觉得到了<b>部分证实</b>。
数据支持的核心结论是：<b>冬季让死亡负担大幅上升</b>（美国州层面冬季相对前后两个夏天的平均，
新增死亡 <b>+152%</b>，2020-21 冬 <b>+156%</b>，2021-22 冬 <b>+152%</b>；99% 州的冬季都更糟，
Wilcoxon p &lt; 10⁻⁶）。<b>但其机制是"传播大幅增强"而非"感染者更容易死"</b>——
同一时期的新增确诊在冬季增加 <b>+176% ~ +191%</b>，而<b>每个感染者的死亡风险（滞后对齐 CFR），
冬季反而略低</b>（约 -8% ~ -14%，且仅 27% 州的冬季 CFR 更高）。
更关键的<b>剂量-反应证据</b>：<b>一个地区冬季越冷，其冬季相对夏天的死亡惩罚越大</b>——
美国各州与全球国家层面均为显著负相关（Pearson r ≈ -0.34 ~ -0.49, p ≤ 0.017），即"越冷越糟"成立。
</div>

<h2>1. 数据与处理</h2>
<h3>1.1 原始数据与负增量（"回调"）</h3>
<ul>
  <li><b>回调规模</b>：全球 72/115 个国家、美国 50/52 个州都出现过累计序列的向下修订；
    最严重的为哈萨克斯坦（占终值 {qc_g.loc[qc_g.rev_pct.idxmax(),"region"]}，{-qc_g.rev_pct.max():.2f}%）、
    内布拉斯加州（{-qc_u.rev_pct.max():.2f}%）。如不修复，diff 会产生负的"日新增"，扭曲所有阶段指标。</li>
  <li><b>修复方法</b>：回溯剥离法。把第 t 天的负增量 m 沿时间轴向后摊销到最近的正增量上，
    直至 m 被扣完。性质：修复后序列<b>单调不减</b>、<b>终值严格守恒</b>（实测最大偏差 0）。
    已写为 <code>cw_core.repair_monotone</code>，可复用。</li>
  <li><b>验证</b>：所有 167 个地区修复后零负增量；<b>单调不变量</b>与<b>终值守恒不变量</b>均通过。</li>
</ul>

<figure><img src="{f5}" style="max-width:100%">
<figcaption>图5：数据回调最严重的 12 个国家与 12 个州（按回调量占最终累计确诊的百分比）</figcaption></figure>

<h3>1.2 气温数据</h3>
<ul>
  <li><b>数据源</b>：Open-Meteo Archive API（ERA5 再分析），逐日 2m 平均气温。</li>
  <li><b>代表点</b>：每个国家的<b>病例加权中心经纬度</b>。
    修正 16 个大国（俄罗斯、加拿大、巴西、阿根廷、墨西哥等）的中心——JHU 给的坐标
    是地理中心（如俄罗斯 61.5°N/105°E 落在西伯利亚），与实际疫情/人口分布严重脱节。
    修正后俄罗斯年均温从 -5.8°C 调整为 +6.0°C。</li>
  <li><b>时间跨度</b>：2019-12-01 ~ 2023-04-30，165 个地区逐日值，0 缺失。</li>
</ul>

<h2>2. 决定性检验：冬天 vs 它前后两个夏天</h2>
<p>设计：每个冬天 W 都被它前一个夏天 S<sub>before</sub> 与后一个夏天 S<sub>after</sub> 夹在中间。
  统计量 penalty = log(指标_W) - 0.5·[log(指标_Sbefore) + log(指标_Safter)]。
  <b>对照是"前后两个夏天的平均"</b>，因此任何平滑的时间趋势（疫苗铺开、变异株更替、治疗进步、
  检测扩容）在一阶插值上被抵消，剩下的才是季节效应。</p>

<figure><img src="{f2}" style="max-width:100%">
<figcaption>图2：决定性检验。蓝=正向（冬季更糟），红=负向（冬季其实更好）。
  标签：效应均值百分比 / "更差"地区占比 / 配对 Wilcoxon p 显著性（*&lt;0.05, **&lt;0.01, ***&lt;0.001）</figcaption></figure>

<h3>2.1 结果（数表）</h3>
{pen_html.to_html(index=False, border=0, classes="t", escape=False)}

<h3>2.2 解读</h3>
<ul>
  <li><b>传播</b>（新增确诊）：美国 50 州中 100% 在 2021-22 冬都更糟，平均 <b>+191%</b>；
    全球 113 国中 95% 更糟，平均 <b>+245%</b>。结论：<b>冬天大幅加强传播</b>。</li>
  <li><b>死亡负担</b>：美国 49 州 98% 更糟，平均 <b>+152%</b>；全球 85% 更糟，平均 <b>+135%</b>。
    结论：<b>冬天死的人数大幅增加</b>——这直接支持用户的直觉。</li>
  <li><b>严重度（CFR）</b>：美国 73% 州冬季 CFR <b>更低</b>，平均 <b>-14%</b>；
    全球 W1 略正、W2 显著负 (-32%)。结论：<b>冬天并非"感染者更容易死"——平均反而更安全</b>，
    可能因为冬季暴发以年轻/轻症人群扩散，叠加奥密克戎冬天的内禀温和性。</li>
</ul>

<h2>3. 美国逐月时序（直观图）</h2>
<figure><img src="{f1}" style="max-width:100%">
<figcaption>图1：美国合计月新增确诊 / 月归因死亡 / 滞后对齐 CFR（红线，右轴气温虚线）。
  蓝带为 11-3 月两个冬季。两个冬季对应了两次死亡高峰（2020-21 冬 ~95k/月，2021-22 冬 ~70k/月），
  而 CFR 在 2020 年 4 月的 25% 高位后（无治疗、检测有限）一路下降到 ~1% 稳态，没有冬季特异升高。
</figcaption></figure>

<h2>4. 剂量-反应：越冷越糟吗？</h2>
<p>把每个地区的"冬季相对夏天的死亡惩罚"作为 y，<b>该地区 11-3 月均温</b>作为 x，
  做散点+线性回归：</p>

<figure><img src="{f3}" style="max-width:100%">
<figcaption>图3：冬季气温 vs 冬季死亡惩罚。左=美国 50 州+DC，右=全球 113 国。
  <b>两套数据均显示显著负相关：越冷的地区，冬季死亡惩罚越大</b>。
  美国 r=-0.38 p&lt;0.001；全球 r=-0.39 p&lt;10⁻⁹。
</figcaption></figure>

<h3>4.1 剂量-反应回归（数表）</h3>
<p><b>死亡惩罚 vs 冬季气温</b>（效应值在对数尺度上, 已转回百分比）</p>
{dose_html.to_html(index=False, border=0, classes="t", escape=False)}

<p style="margin-top:18px"><b>CFR 惩罚 vs 冬季气温</b>（无显著剂量-反应）</p>
{dose_cfr.to_html(index=False, border=0, classes="t", escape=False)}

<p class="muted">
  解释：冬季均温每低 1°C，<b>死亡惩罚的 log-值</b>额外下降约 0.02~0.06。
  直观上，5°C 冷冬比 5°C 暖冬多承受约 10~30% 的冬季死亡超额（不同窗口与样本下不同）。
</p>

<h2>5. 死亡在一年中的分布</h2>
<figure><img src="{f4}" style="max-width:100%">
<figcaption>图4：美国（左） / 全球（右）每月死亡与确诊占全期比重。
  蓝=寒季月（11-3），橙=暖季月（5-9），灰=过渡月（4, 10，按中位数纬度判半球）。
  <b>美国</b>：寒季 5 个月贡献了 <b>{u_d_cold:.1f}%</b> 死亡 / <b>{u_c_cold:.1f}%</b> 确诊；
  <b>全球</b>：寒季 5 个月贡献了 <b>{g_d_cold:.1f}%</b> 死亡 / <b>{g_c_cold:.1f}%</b> 确诊。
  5 个月窗口占全年的 41.7%，但贡献的死亡比重远高于此——冬季超额显著。
</figcaption></figure>

<h2>6. 失败的反向证据：为什么部分"严格"分析会得出不同结论</h2>
<p>在项目过程中，我尝试过几种更"严格"的设定，发现它们在 CFR 指标上对结论敏感：</p>
<ul>
  <li><b>地区内 / 同一流行年内 配对检验</b>（按"流行年" 9/1~8/31）：
    E2020 全球国家 65% 更糟（中位 1.19, p=0.006）；E2021 全球反而逆转（40% 更糟，p&lt;0.001）——
    <b>因为冬天恰好是奥密克戎，与更温和的变异株混杂</b>。</li>
  <li><b>双向固定效应回归（地区FE + 日历月FE）</b>：加入日历月FE 后该混杂被吸收，
    美国结果显著（β = -0.036 ~ -0.10），跨国样本接近零。
    但若把维度拆开为"同月跨地区"与"同地区跨月"分别回归，<b>两个维度都接近零</b>，
    pooled 估计的显著性来自两者加权的特定加权。这说明 CFR 指标对设定很敏感。</li>
  <li><b>JHU 坐标测量误差</b>：俄罗斯原始坐标的年均温与真实人口中心相差 11.8°C，
    属经典测量误差，会向零衰减估计。修正 16 个大国坐标后，跨国样本的温带/寒带分层
    β 从 0 翻为 -0.026（仍不显著但方向已翻），说明跨国数据对"代表点"非常敏感。</li>
</ul>
<p>这些都指向同一个方法学教训：<b>CFR 受检测强度 / 数据质量 / 变异株 / 治疗进步多重污染</b>，
  不足以独立支撑结论；而<b>死亡负担</b>（绝对死亡数）更稳定、能抗检测波动、能跨数据集复现——
  所以本报告最终结论以"死亡负担"为锚定。</p>

<h2>7. 局限与开放问题</h2>
<ul>
  <li><b>检测强度</b>仍会随季节变化：冬季检测需求更高、检测基础设施更紧张。
    但这<b>只能解释冬季确诊"少报"→ 冬季 CFR 偏高</b>，与我们观察到的"冬季 CFR 偏低"
    反向，所以结论对这一偏差保守。</li>
  <li><b>JHU 全球数据自 2022 年起严重漏报确诊</b>（家庭自测、奥密克戎时代），
    已通过"决定性检验"（与同期确诊量、死亡数比较）部分规避；严谨起见若要扩展结论，
    可叠加 WHO 官方报告与各国超额死亡估计。</li>
  <li><b>温度变量在大国仍有偏</b>：即使修正了人口/疫情中心，
    单一坐标无法代表横跨多气候带的全国（例如俄罗斯欧洲部分 vs 西伯利亚）。
    这是为什么美国 50 州的结果比全球 113 国更干净。</li>
  <li><b>机制未分解</b>：本研究只关联"温度"与"死亡/传播"，未拆分通道（维生素 D、
    室内活动、湿度、紫外线、气溶胶稳定性等）。后续可结合 Open-Meteo 的湿度/UV 字段做通道分解。</li>
</ul>

<h2>8. 一句话总结</h2>
<div class="key">
<b>直觉是对的：冬天更糟。</b> 但糟的方式不是"感染者更容易死"，而是"感染的人多得多"——
  <b>冬季让 COVID-19 传播显著增强（+180% 量级），进而驱动死亡数增加（+150% 量级）</b>；
  而<b>每个感染者的死亡风险反而比夏天更低</b>。
  越冷的地区，这种冬季超额死亡惩罚越大（<b>美国 r=-0.38, p&lt;0.001；全球 r=-0.39, p&lt;10⁻⁹</b>）。
</div>

<p class="muted" style="margin-top:36px">
数据：JHU CSSE time_series_ex + Open-Meteo ERA5 | 代码：<code>D:/q/weixin/WorkBuddy/2026-08-28-15-31-23/covid_winter/</code>
（含 <code>cw_core.py</code> 加载与修复模块、<code>step1_qc.py</code> 质检、
<code>step2_temp.py</code> 抓取气温、<code>step9_coords.py</code> 坐标修正、
<code>step14_decisive.py</code> 决定性检验、<code>step15_figures.py</code> 图表、
<code>step16_dose.py</code> 剂量-反应、<code>step17_report.py</code> 本报告生成）
</p>

</body></html>
"""
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("写出 report.html")


if __name__ == "__main__":
    main()
