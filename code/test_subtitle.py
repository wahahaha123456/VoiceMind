# -*- coding: utf-8 -*-
"""字幕工具离线回归:断句 / 热词校正 / SRT 生成(不加载模型)。"""
import sys, time
sys.path.insert(0, r"C:\Users\iiiis\Desktop\FunASR\code")
t0 = time.time()
import app
print("import app 耗时 %.1fs, 路由数 %d" % (time.time() - t0, len(app.app.routes)))

cases = [
    ("就是F幺零这个排球队的蓝心妹现在怎么样", "篮球队 蓝心湄 剪映",
     "就是F幺零这个篮球队的蓝心湄现在怎么样"),
    ("我们在剪映里打开工程", "剪映", "我们在剪映里打开工程"),
    ("蓝心湄的演出很好", "蓝心湄", "蓝心湄的演出很好"),
    ("这个剪映版本不错达摩院的模型也好", "剪映 达摩院", "这个剪映版本不错达摩院的模型也好"),
    ("蓝心的天很蓝", "蓝心湄", "蓝心的天很蓝"),
    ("他们的用户体验做得好", "剪映", "他们的用户体验做得好"),
]
for t, hw, want in cases:
    r, f = app._apply_hotwords(t, hw)
    mark = "OK  " if r == want else "FAIL"
    print("%s %-24s -> %-24s %s" % (mark, t, r, f or "(无校正)"))
    assert r == want, (t, r, want)
print("热词: 全部通过")

srt_cases = [
    "那个我再问一下，就是咱们院里这个篮球队还有足球队这个训练情况，最近练得怎么样？",
    "好大家上午好这个一直以来咱们商场的销售情况都很好特别是最近营业额有很大的提升",
    "短句",
    "今天天气不错我们去公园散步然后顺便买点东西回家做午饭",
]
for t in srt_cases:
    ls = app.split_subtitle(t, 15)
    assert all(len(x) <= 15 for x in ls), (t, ls)
    assert not any(len(x) == 1 and x in "。！？，" for x in ls), ("孤儿标点", ls)
    print("  " + " | ".join(ls))
print("断句: 全部通过(每行≤15字,无孤儿标点)")

segs = [{"text": srt_cases[0], "start": 0, "end": 12000}]
srt = app.generate_srt(segs, offset=1.5, max_len=15)
assert srt.startswith("1\n00:00:01,500 -->"), srt
print("SRT: 偏移校正通过")
print(srt)
