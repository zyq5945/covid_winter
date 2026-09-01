# -*- coding: utf-8 -*-
"""减少论文中的加粗：只保留结构性引导语与极少数短强调，其余全部去粗。"""
import re, sys, shutil

sys.stdout.reconfigure(encoding='utf-8')
SRC = '冬天真的更致命吗.md'
shutil.copy(SRC, SRC + '.bak_bold')

text = open(SRC, encoding='utf-8').read()
lines = text.split('\n')

# 保留规则
RE_POINT = re.compile(r'^(第[一二三四五六七八九十]+|[一二三四五六七八九十]+)[，、,]\s*\S')  # 第一，/ 一，
KEEP_SHORT = {'不', '不是', '也不', '并没有', '没有', '更低', '下降', '待测'}

kept, removed = [], 0


def decide(content, is_table, at_line_start, is_list_leading):
    """返回 True 表示保留加粗。"""
    if is_table:
        return False
    c = content.strip()
    # A. 分点引导句：第一，…… / 一，……
    if RE_POINT.match(c):
        return True
    # B. 列表项开头的术语
    if is_list_leading and len(c) <= 20:
        return True
    # C. 段首（含引用块）的引导句：较短且以句号结尾
    if at_line_start and len(c) <= 40 and c.endswith(('。', '：', ':')):
        return True
    # D. 极短的强否定/反转强调
    if c in KEEP_SHORT:
        return True
    return False


out_lines = []
for line in lines:
    if '**' not in line:
        out_lines.append(line)
        continue
    stripped = line.lstrip()
    is_table = stripped.startswith('|')
    prefix_len = len(line) - len(stripped)
    m = re.match(r'^(>\s*)', stripped)
    if m:
        core = stripped[len(m.group(1)):]
        body_start = prefix_len + len(m.group(1))
    else:
        core = stripped
        body_start = prefix_len
    is_list = bool(re.match(r'^([-*]|\d+\.)\s', core))

    new_line = line
    spans = list(re.finditer(r'\*\*(.+?)\*\*', line, flags=re.S))
    # 从后往前替换，避免位置偏移
    for sp in reversed(spans):
        content = sp.group(1)
        pos = sp.start()
        at_line_start = (pos == body_start)
        # 列表项开头术语：该 span 紧跟在 "- " / "1. " 之后
        head = line[body_start:pos]
        is_list_leading = bool(re.fullmatch(r'([-*]|\d+\.)\s*', head))
        if decide(content, is_table, at_line_start, is_list_leading):
            kept.append(content.strip()[:60])
        else:
            removed += 1
            new_line = new_line[:sp.start()] + content + new_line[sp.end():]
    out_lines.append(new_line)

new_text = '\n'.join(out_lines)
# 清理可能残留的空加粗
new_text = new_text.replace('****', '')
open(SRC, 'w', encoding='utf-8', newline='\n').write(new_text)

print(f'原有加粗片段: {removed + len(kept)}')
print(f'去掉: {removed}')
print(f'保留: {len(kept)}')
print('\n--- 保留清单 ---')
for i, k in enumerate(kept, 1):
    print(f'{i:3d}. {k}')
print('\n残留 ** 数量:', new_text.count('**'))
