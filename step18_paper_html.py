"""把论文 Markdown 转成排版好的 HTML（插图嵌入）"""
from __future__ import annotations
import re, io, base64, os


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r'<code class="ic">\1</code>', s)
    return s


def convert(md: str, figures: dict) -> str:
    """figures: {锚点关键字: (路径, 图注)}"""
    lines = md.split("\n")
    out, i = [], 0
    n = len(lines)

    def flush_paragraph(buf):
        if buf:
            out.append(f"<p>{inline(' '.join(buf))}</p>")
            buf.clear()

    buf = []
    while i < n:
        ln = lines[i].rstrip()

        # 插图锚点
        hit = None
        for key, (fp, cap) in figures.items():
            if key in ln and os.path.exists(fp):
                hit = (fp, cap); break
        if hit:
            flush_paragraph(buf)
            out.append(f'<figure><img src="{data_uri(hit[0])}" alt="{hit[1]}">'
                       f'<figcaption>{hit[1]}</figcaption></figure>')

        if ln.startswith("### "):
            flush_paragraph(buf)
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            flush_paragraph(buf)
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            flush_paragraph(buf)
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.strip() == "---":
            flush_paragraph(buf)
            out.append("<hr>")
        elif ln.startswith("|"):
            flush_paragraph(buf)
            block = []
            while i < n and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            out.append(table_html(block))
            continue
        elif ln.startswith("> "):
            flush_paragraph(buf)
            out.append(f'<blockquote>{inline(ln[2:])}</blockquote>')
        elif re.match(r"^\s*[-*] ", ln):
            flush_paragraph(buf)
            items = []
            while i < n and re.match(r"^\s*[-*] ", lines[i]):
                items.append(f"<li>{inline(re.sub(r'^\s*[-*] ', '', lines[i].strip()))}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\s*\d+\. ", ln):
            flush_paragraph(buf)
            items = []
            while i < n and re.match(r"^\s*\d+\. ", lines[i]):
                items.append(f"<li>{inline(re.sub(r'^\s*\d+\. ', '', lines[i].strip()))}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        elif ln.strip() == "":
            flush_paragraph(buf)
        elif ln.startswith("*"):
            buf.append(ln)
        else:
            buf.append(ln)
        i += 1
    flush_paragraph(buf)
    return "\n".join(out)


def table_html(block):
    rows = []
    for r in block:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    head, body = rows[0], rows[2:] if len(rows) > 2 else []
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


CSS = """
body{font-family:"Georgia","Songti SC","SimSun",serif;line-height:1.95;
     max-width:860px;margin:0 auto;padding:48px 28px 90px;color:#24292f;
     font-size:17px;letter-spacing:.01em}
h1{font-family:"Microsoft YaHei",sans-serif;font-size:32px;line-height:1.35;
   border-bottom:3px solid #2b6cb0;padding-bottom:10px;margin-bottom:4px}
h2{font-family:"Microsoft YaHei",sans-serif;font-size:22px;color:#1a365d;
   margin-top:46px;padding-left:12px;border-left:5px solid #2b6cb0}
h3{font-family:"Microsoft YaHei",sans-serif;font-size:17.5px;color:#2c5282;
   margin-top:32px;border-bottom:1px dashed #cbd5e0;padding-bottom:4px}
p{margin:14px 0;text-align:justify}
strong{color:#9b2c2c;font-weight:700}
hr{border:0;border-top:1px solid #e2e8f0;margin:38px 0}
ul,ol{padding-left:26px;margin:14px 0}
li{margin:7px 0}
table{border-collapse:collapse;width:100%;margin:20px 0;font-size:15px;
      font-family:"Microsoft YaHei",sans-serif}
th{background:#edf2f7;padding:9px 12px;text-align:left;border:1px solid #cbd5e0;
   font-weight:600;color:#1a365d}
td{padding:8px 12px;border:1px solid #e2e8f0;font-variant-numeric:tabular-nums}
tr:nth-child(even) td{background:#f7fafc}
figure{margin:30px 0;text-align:center}
figure img{max-width:100%;border:1px solid #e2e8f0;border-radius:4px}
figcaption{font-family:"Microsoft YaHei",sans-serif;font-size:13.5px;
           color:#4a5568;margin-top:8px;text-align:left;line-height:1.6}
blockquote{margin:18px 0;padding:12px 18px;background:#f7fafc;
           border-left:4px solid #90cdf4;color:#2d3748;font-size:16px}
code.ic{background:#edf2f7;padding:1px 6px;border-radius:3px;font-size:14px;
        font-family:Consolas,monospace;color:#9b2c2c}
.sub{font-family:"Microsoft YaHei",sans-serif;color:#4a5568;font-size:16px;
     margin:6px 0 26px;font-weight:400}
@media(max-width:640px){body{padding:24px 16px;font-size:16px}h1{font-size:26px}}
"""


if __name__ == "__main__":
    md = io.open("冬天真的更致命吗.md", encoding="utf-8").read()
    figures = {
        "## 四、结果": ("figures/fig2_penalty.png",
                   "图 1　决定性检验：每个冬天相对它前后两个夏天平均的变化。蓝色为正向（冬季更糟），红色为负向（冬季其实更好）。标签依次为效应幅度、“更差”地区占比与显著性。"),
        "### 4.1 冬天，死的人确实多得多": ("figures/fig4_monthshare.png",
                   "图 2　死亡与确诊在一年中的分布。上排为死亡占比，下排为确诊占比；左列美国，右列全球。蓝色为寒季月（11—3 月），橙色为暖季月（5—9 月），灰色为过渡月。"),
        "### 4.3 越冷的地方，冬天越吃亏": ("figures/fig3_scatter.png",
                   "图 3　剂量—反应：横轴是各地区冬季（11—3 月）平均气温，纵轴是冬季相对夏天的死亡超额。左为美国 52 个地区，右为全球。两套数据都呈显著负相关。"),
        "## 七、所以呢": ("figures/fig1_us_monthly.png",
                   "图 4　美国合计的逐月走势：新增确诊、归因死亡、滞后对齐病死率，叠加月均气温虚线。蓝色带为两个冬季。"),
        "**修正后把两个半球合起来算**": ("figures/fig6_hemisphere.png",
                   "图 5　南北半球对比：左为各窗口的死亡超额（含双侧 / 单侧对照分解），右为宏观剂量—反应（冬夏温差 vs 死亡惩罚）。"),
    }
    body = convert(md, figures)
    html = (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>冬天真的更致命吗</title><style>{CSS}</style></head><body>{body}</body></html>')
    io.open("冬天真的更致命吗.html", "w", encoding="utf-8").write(html)
    print("写出 冬天真的更致命吗.html 大小:", len(html))
