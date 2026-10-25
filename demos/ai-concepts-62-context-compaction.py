#!/usr/bin/env python3
"""第 62 篇 demo：会话压缩（compaction）到底丢了什么。

真实 LLM 实测：deepseek-chat（api.deepseek.com，temperature=0）。
不是 GPU 实测、不是多模型对比：一个模型、一段合成会话、关键词判分。

会话 24 轮 ≈ 12k 字；按 400 / 150 / 60 字三档预算压缩（约 30x / 80x / 200x），
再对 21 条事实 + 3 条「为什么」探测，看存活率怎么随预算塌。

跑法：python3 run_with_env.py 62-context-compaction.py
"""
import json
import os
import time
import urllib.request

API = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-chat'
KEY = os.environ['DEEPSEEK_API_KEY']


def chat(messages, max_tokens=800, temperature=0.0):
    body = json.dumps({'model': MODEL, 'messages': messages,
                       'temperature': temperature, 'max_tokens': max_tokens}).encode()
    req = urllib.request.Request(API, data=body, headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + KEY})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            return d['choices'][0]['message']['content'].strip(), d['usage']
        except Exception as e:
            if attempt == 3:
                raise
            print(f'  ! retry {attempt + 1}: {e}')
            time.sleep(3)


# ---------------------------------------------------------------- 1. 合成会话
TURNS = [
    (1, '新项目开工，先把几条铁律钉死，后面别再改。第一，所有对外报价一律用人民币，绝对不许出现美元；'
        '之前给一个海外客户报价用了美元，对方财务一走汇率流程拖了三周，项目直接黄了，所以这条是硬要求，'
        '不是偏好。第二，文章正文里不许放外部网站链接，除了我们自己的 demo 仓库以外——读者点了外链跳走，'
        '跳出率高得离谱，这个数据我看过，没必要再试一次。别人的工作写名字就行，不给 URL。',
     '人民币计价、正文禁外链（自有仓库除外）。这两条我写进 AGENTS.md 顶部。'),
    (2, '中午我先去楼下吃面，不回来吃了，下午那版我看了三遍，排版有点松，行距和段距都不对，'
        '中文长段落读起来累。另外页脚那个版权年份还是硬编码的，明年又得改一次。',
     '行距段距我收紧到 1.85 和 1.2em；版权年份改成动态取。面馆那家人多，早点去。'),
    (3, '训练配置定一下：batch size 就 48，单卡显存刚好卡在 90% 出头，再大一点直接 OOM，'
        '我试过 56，跑到第二个 epoch 就崩了。旧的那套 32 是给 8 卡集群写的，别拿来凑数，'
        '梯度累积步数不一样，混用会把有效 batch 算错。',
     'batch size = 48 / 单卡；旧的 32 配置标 deprecated，不进默认配置。'),
    (4, '向量库这事我想了整整两天，跟你交个底。Qdrant 我玩过一轮，检索效果确实好，但运维太重，'
        '得单独盯一个服务、单独做备份、单独升版本，出问题还得自己查；我们一共三个人。最后定 pgvector，'
        '直接用现有 Postgres，不引第二个组件，备份和监控复用已有的。这个不再改了，就 pgvector。',
     'pgvector 定稿，Qdrant 方案归档到 docs/decisions。'),
    (5, '署名这块有个硬要求：统一写「赛博阿漆」，不许再出现「阿漆」「漆哥」这类简称。'
        '我做过一次小调查，品牌名不一致读者真的记不住，两个名字在两个平台传，等于把流量劈成两半。'
        '简介也一起换，旧简介里还是老名字，搜出来第一屏就是错的。这个改起来麻烦，但必须改。',
     '全文替换为「赛博阿漆」，简介、meta、og 标签一起换。'),
    (6, '部署我换平台了，国内那家不用了，改 Fly.io 新加坡区。原因是东南亚读者占了三分之一，'
        '从国内机房绕一圈过去延迟 300ms 起，首屏看着像卡死。磁盘 10G 够用，静态站又不吃存储，'
        'CDN 那层我另配。区域别选错，选悉尼会绕更远。',
     'Fly.io 新加坡区，磁盘 10G，CDN 另配。'),
    (7, '学习率用 3e-5，warmup 3%，这个是从上一版实验里一格一格试出来的，别再动。'
        '外面开始下雨了，窗户得关一下，键盘进水过一次。另外那台旧笔记本的风扇声音越来越大，'
        '回头得清灰。',
     '3e-5 + 3% warmup 锁定，不参与调参。') ,
    (8, 'UI 里一律不许用 emoji，按钮图标统一用 // 这种等宽符号，难度徽章写 L1/L2/L3。'
        '但概念卡片正文里的 emoji 小节保留——那属于写作风格，是给读者分段用的，不是 UI 装饰。'
        '这两件事别混，上一版有人把正文 emoji 一起清了，读者直接说「看着像文档」。',
     'UI 禁 emoji；正文保留。这条写进设计规范。'),
    (9, 'GitHub 那边超时设 6 秒，之前设 30 秒挂死过两次，一次在凌晨，任务挂到早上我才发现。'
        '单请求短超时 + 循环重试，别跑长挂起循环，那个会把配额耗光还什么都拿不到。',
     '超时 6 秒 + 重试；不写长挂起循环。'),
    (10, '构建必须用 node20，node18 会被 Astro 直接拒掉，报 Node.js v18.20.4 is not supported。'
         '这个坑我踩过了，当时以为是代码错，查了两小时。写进 README，别让下一个人再花两小时。',
     'node20 固定；README 记坑。'),
    (11, '我猫又跳桌子上了，把线全扒了，显示器黑了三秒。算了先说正事：KV cache 上限设 8G，'
         '超了就排队，别让它吃满整机内存把其他服务挤死。这台机器一共才 16G，之前吃满过一次，'
         'SSH 都连不上。',
     'KV cache 上限 8G，超限排队。'),
    (12, '索引页那块我想清楚了：不写 seriesOrder。写上就会被 PostNav 算进期数，期数虚高，'
         '读者一看 60 期以为很多，其实索引本身不算一期。索引页只要 series + slug 就够了，'
         '这样进度条也不会把索引算成一篇。',
     '索引页只留 series + slug。'),
    (13, '备份保留 7 天，每天一份完整还原包，凌晨一点跑。旧的超过 7 天直接删，别攒着占空间，'
         '上次攒了两个月，磁盘直接满，写日志都失败。完整包的意思是可还原，不是只备配置。',
     '备份 7 天滚动，01:00 跑。'),
    (14, '调用模型换成 deepseek-v4-flash，老的那个响应太慢，写文章这种活等不起，'
         '一篇文章要等四十秒才吐第一个字。API key 还是从 .env 读，别写进代码，'
         '更别写进命令行，命令行会进日志。',
     '模型换 deepseek-v4-flash；key 只从 .env 读。'),
    (15, '节奏上定死一条：每周最多发三篇，多了质量就掉，这个我自己有体感。'
         '这周先把压缩那篇写完，别的排队；不要因为「反正有模板」就多产几篇。',
     '每周 ≤3 篇。'),
    (16, '采样率统一 16k，板子和服务器两边都是。24k 会被重采样，VAD 直接失效，'
         '这个我实测过：同一句话，24k 过去检测不到句尾，16k 正常。所以别在中间某一层偷偷升采样。',
     '两端 16k 一致，禁止中间层升采样。'),
    (17, '地铁今天晚点二十分钟，我在站台把稿子改完了。接着说：发布流程有个硬约束，'
         '必须我手动确认一次才能上线，不许全自动。全自动发过一次，标题错了半小时才发现，'
         '删了重发，RSS 里留了两条重复。',
     '发布前人工确认，不自动上线。'),
    (18, '服务端口定 8000 给 WS，8003 走 OTA，别再混用。文档里写清楚，新人上手别猜。'
         '端口冲突这种问题排查起来最浪费时间，日志里还不报，就是连不上。',
     '8000 WS / 8003 OTA，文档写明。'),
    (19, '报告这事我态度很明确：QQ 只发一句话纯文本，不许出现表格线和等宽块，'
         '手机上看那种排版是灾难，横着竖着都对齐不了。完整报告走邮件，PDF 附件。',
     'QQ 一句话纯文本；完整报告走邮件。'),
    (20, '摄像头和麦克风这两件事，没我明说「启动」就不许碰，不许自己抓帧，也不许自触发定时任务。'
         '这条是底线，不是技术限制；硬件在那儿不等于可以使用，权限边界必须由人给。',
     '感官硬件需明示授权，禁止自触发。'),
    (21, '预算表我算了一遍，一共 12 万，其中硬件 3 万、外包 4 万、剩下的是算力。'
         '报价单按人民币整百取整，别出现 88.37 这种数，客户看着不专业，财务对账也烦。',
     '预算 12 万；报价整百取整。'),
    (22, '这套设计我否了，纯装饰的东西太多。HUD 每个元素都得有功能含义：呼吸方点表示站点在线，'
         '四角括线表示可交互边界，指针高光表示焦点。没有含义的扫描线、粒子背景，一律不做。',
     '设计只保留有功能含义的元素。'),
    (23, '每周复盘放周日，一次批量清，别天天问我清不清，问多了我也烦。'
         '清理范围包括旧包、备份、缓存、旧日志，先列清单和总量，确认一次再动手。',
     '周日批量清理，先列清单再确认。'),
    (24, '最后确认两件事：品牌名是「赛博阿漆」，报价是人民币。这两条最容易被后面接手的人改坏，'
         '写进 AGENTS.md 置顶，谁改谁负责。今天就到这儿，明天继续。',
     '两条写进 AGENTS.md 置顶。'),
]

FACTS = [
    dict(id='c1', kind='C', turn=1, q='对外报价用什么货币？有什么禁止项？', req=['人民币']),
    dict(id='c2', kind='C', turn=1, q='文章正文能不能放外部链接？', req=['不能', '不许', '禁止', '不要', '不写', '除外']),
    dict(id='c3', kind='C', turn=5, q='品牌名该怎么写？有什么不允许的？', req=['赛博阿漆']),
    dict(id='c4', kind='C', turn=8, q='UI 里能不能用 emoji？正文呢？', req=['不许', '不能', '禁止', '禁', '不要']),
    dict(id='c5', kind='C', turn=17, q='发布上线前需要什么动作？', req=['手动', '确认', '人工']),
    dict(id='c6', kind='C', turn=19, q='QQ 通知的格式要求是什么？', req=['一句话', '纯文本', '不要表格', '禁']),
    dict(id='c7', kind='C', turn=20, q='摄像头和麦克风的使用前提是什么？', req=['明说', '授权', '启动', '不许']),
    dict(id='c8', kind='C', turn=24, q='哪两条要写进 AGENTS.md 置顶？', req=['赛博阿漆', '人民币']),
    dict(id='d1', kind='D', turn=4, q='向量库最后定的是哪个？', req=['pgvector']),
    dict(id='d2', kind='D', turn=6, q='部署在哪个平台和区域？', req=['新加坡']),
    dict(id='d3', kind='D', turn=10, q='构建必须用哪个 node 版本？', req=['20']),
    dict(id='d4', kind='D', turn=12, q='索引页要不要写 seriesOrder？', req=['不写', '不要写', '不需要']),
    dict(id='d5', kind='D', turn=14, q='调用模型换成了哪个？', req=['v4-flash', 'v4']),
    dict(id='d6', kind='D', turn=18, q='WS 和 OTA 各用哪个端口？', req=['8000']),
    dict(id='n1', kind='N', turn=3, q='batch size 定成多少？', req=['48']),
    dict(id='n2', kind='N', turn=7, q='学习率用的多少？', req=['3e-5', '3e-05', '0.00003', '3×10']),
    dict(id='n3', kind='N', turn=9, q='GitHub 单请求超时设几秒？', req=['6']),
    dict(id='n4', kind='N', turn=11, q='KV cache 上限设多少？', req=['8']),
    dict(id='n5', kind='N', turn=16, q='采样率定成多少？', req=['16k', '16 k', '16000']),
    dict(id='n6', kind='N', turn=21, q='总预算多少？报价怎么取整？', req=['12']),
    dict(id='f1', kind='F', turn=2, q='午饭吃的是什么？', req=['面']),
    dict(id='f2', kind='F', turn=7, q='那天外面天气怎么样？', req=['雨']),
    dict(id='f3', kind='F', turn=11, q='猫干了什么？', req=['桌', '线']),
    dict(id='f4', kind='F', turn=17, q='地铁出了什么事？', req=['晚点', '延误']),
]

# 「为什么」探测：答案里的理由，被压缩掉的概率远高于答案本身
WHY = [
    dict(id='w1', kind='W', q='为什么品牌名必须统一？', req=['记不住', '流量', '劈', '调查']),
    dict(id='w2', kind='W', q='为什么放弃 Qdrant 选 pgvector？', req=['运维', '重', '服务', '组件']),
    dict(id='w3', kind='W', q='为什么采样率必须是 16k？', req=['重采样', 'VAD', '句尾', '检测']),
]

TRANSCRIPT = '\n'.join(f'【第{t}轮 · 用户】{u}\n【第{t}轮 · 助手】{a}' for t, u, a in TURNS)
BUDGETS = [400, 150, 60]


def compact_prompt(budget):
    return (f'把下面这段对话压缩成不超过 {budget} 字的中文摘要，压缩掉重复与寒暄，'
            '保留后续任务需要的信息。只输出摘要，不要解释。')


def hit(req, answer):
    a = answer.replace(' ', '')
    return any(r.replace(' ', '') in a for r in req)


def probe(fact, context, label):
    msg = ('下面是一段项目会话记录。只依据记录回答，记录里没写的就回答「未提及」。'
           f'答案控制在 30 字以内，不要解释。\n\n=== 记录开始 ===\n{context}\n'
           f'=== 记录结束 ===\n\n问题：{fact["q"]}')
    ans, _ = chat([{'role': 'user', 'content': msg}], max_tokens=60)
    ok = hit(fact['req'], ans)
    print(f'  [{label}] {fact["id"]}: {"OK  " if ok else "MISS"} {ans[:46]}')
    return dict(fact=fact['id'], kind=fact['kind'], cond=label, ok=ok, ans=ans[:80])


def main():
    for f in FACTS:
        if f['kind'] != 'F':
            assert any(r in TRANSCRIPT for r in f['req']), f['id']
    print('轮数', len(TURNS), '字符数', len(TRANSCRIPT))

    conds = {}
    for b in BUDGETS:
        s, u = chat([{'role': 'system', 'content': compact_prompt(b)},
                     {'role': 'user', 'content': TRANSCRIPT}], max_tokens=900)
        conds[f'S{b}'] = s
        print(f'预算 {b} 字 → 实际 {len(s)} 字 / {u["completion_tokens"]} tok'
              f'（压缩 {len(TRANSCRIPT) / max(len(s), 1):.0f}x）')
    print('\n--- 400 字版 ---\n' + conds['S400'])
    print('\n--- 150 字版 ---\n' + conds['S150'])
    print('\n--- 60 字版 ---\n' + conds['S60'] + '\n')

    allc = [('FULL', TRANSCRIPT)] + [(k, v) for k, v in conds.items()]
    rows = []
    for label, ctx in allc:
        print(f'=== {label} ===')
        for f in FACTS:
            rows.append(probe(f, ctx, label))
        for f in WHY:
            rows.append(probe(f, ctx, label))

    print('\n类型 × 条件（存活/总数）')
    print(f'{"":>6}' + ''.join(f'{c:>10}' for c, _ in allc))
    for k in ['C', 'D', 'N', 'W', 'F']:
        line = f'{k:>6}'
        for c, _ in allc:
            sel = [r for r in rows if r['kind'] == k and r['cond'] == c]
            line += f'{sum(x["ok"] for x in sel):>7}/{len(sel):<3}'
        print(line)
    print('\n合计：', {c: f'{sum(r["ok"] for r in rows if r["cond"] == c)}/{len(FACTS) + len(WHY)}'
                       for c, _ in allc})

    with open('62-context-compaction.json', 'w', encoding='utf-8') as fp:
        json.dump(dict(summaries=conds, conds=[c for c, _ in allc], rows=rows),
                  fp, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
