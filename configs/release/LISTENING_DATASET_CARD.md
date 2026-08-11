---
license: cc-by-nc-4.0
language:
  - zh
  - en
pretty_name: 人工盲听（MOS 与情绪标签核验）
task_categories:
  - text-to-speech
tags:
  - speech
  - mos
  - listening-test
  - chinese
size_categories:
  - n<1K
---

# 人工盲听

三组盲听材料，加上揭盲用的答案表。四个人填回来的评分原件不在这个仓，在代码仓库的
`artifacts/listening_returned/`。这是 mindsurf-omni 唯一一把人耳尺子——其余的音质
数字都出自自动指标，而自动指标有它自己的盲点。

发包时每包切成三份，实际来填的是四个人，所以有两个人拿的是同一份（顺序也相同）。
算人与人之间的一致率时，那一对是同一批刺激直接对比。

## 下哪几个文件夹

三个包，每包按人分成三份，各拿各的那一份：

| 包 | 听什么 | 每人几条 |
| --- | --- | ---: |
| `listening_models` | 两套模型说的同一批话，打自然度 | 64 |
| `listening_synthesiser` | 两个合成器念的同一批话，打自然度 | 64 |
| `listening_emotion` | 语料里的真人音频，写下你听到的情绪 | 55 |

你是 rater2，就下这三个：`listening_models/rater2/`、`listening_synthesiser/rater2/`、
`listening_emotion/rater2/`。

每个文件夹里是 `raterN.xlsx`（列宽、冻结表头和下拉都设好了）、`raterN.csv`
（同一张表的纯文本版，两个填哪个都行，交回一个），以及从 `001.wav` 开始编号的音频。
编号就是表里的行号，从头往下听、往下填，不用自己找文件。

每一列填什么、怎么打分，看每个包根目录下的 `README.txt`。三个包的规则不一样，
别拿一个包的标准去填另一个。

## 两件容易踩的事

**三个人的编号是错开的。** 同一条音频在不同人的表里编号不同，这是故意的，所以不要互相对编号。

**每个包里混了重复条目**，用来看同一个人前后给分是否一致。听到像是听过的，照常打分。

## 答案表

生成这些包的脚本会同时写一份 `key.json`，记着每条音频出自哪个系统、
或者那条语料原本被标成什么情绪。包里的 `README.txt` 写着「不要打开 key.json」，
发包的时候我们干脆没有传，评分期间这个仓库和代码仓库里都没有它。

**2026-08-05 十二份表全部收齐，答案表连同评分一起公开。**
现在每个包的根目录下都有：`listening_models/key.json`、
`listening_synthesiser/key.json`、`listening_emotion/key.json`。

评分原件和算分脚本在代码仓库（`artifacts/listening_returned/`、
`scripts/listening_test.py`）。三样凑齐可以自己把 MOS 重算一遍：

```bash
python scripts/listening_test.py score \
  --pack artifacts/listening_returned/listening_models \
  --key artifacts/listening_models/key.json
```

`score` 只适用于两个打分的包（听合成器那包把命令里的 models 换成 synthesiser）。
情绪包填的是情绪选项不是 1–5 分，拿它跑这条命令会以「the mos_1_to_5 column is empty」
退出；那 31–42% / 41–65% 要自己拿 `listening_returned/listening_emotion/` 的选项
对 `listening_emotion/key.json` 比，目前没有现成的子命令。

它会打出两个平均值，差别不是舍入。包里混了重复片段，所以每条片段被打分的次数不等
（有的五次、有的四次）；按片段先取均值再平均，会把听的人少的片段和听的人多的算成一样重。
**我们对外报的是逐条评分那个平均**，每条打分只数一次——我们平均的是人的判断，不是片段的质量。

## 这批评分后来得出了什么

四个人填完，三个包各给出一条结论。

**`listening_emotion` 解释了一轮训练为什么零效应。** 情绪条件化那一轮我们跑过了：
主判据音高零效应，四条情绪臂对中性全部判无法区分，而同批的对照臂正常，
所以是真的没效果，不是仪器读不出来（逐条读数在代码仓库 `artifacts/emotion-conditioned/`）。

这批盲听补上的是第二重否定：人对 `emotion2vec` 自动标签的一致率是 31–42%，而人对人是
41–65%。两个数都低说明这个任务本来就难；但人彼此对得上、却都对不上标签，
那是标签系统性偏离人耳。也就是说「再训一轮」也解决不了，前提本身不成立。

**`listening_models` 给出了那把人耳尺子。** 打分的三个系统在 `key.json` 里的标签是
`graft`（我们的嫁接臂）、`official`（上游 `sft_omni`）、`edge-tts`（参照上限），各 20 条。

**`graft` 不是权重仓里发布的 `sft_merge`。** 两者的关系是量过的：`sft_merge` 的 Talker
和音频投影层与父逐位相同，且它对父的错字率与音质分配对判定都是无法区分——拿这个分数
代表成品，靠的是这条链。

`graft` 和 `official` 都是 2.738，到小数点后三位相同，缺陷标记 43 对 42；
同一个包里的 edge-tts 是 4.023。在这之前「与官方无法区分」只有 UTMOS 一把自动尺子撑着，
现在有一把不可能继承 UTMOS 盲点的仪器落在了同一处。

**这个 panel 不作门控判定**，算分脚本自己会拒绝出判词：每系统 20 条，要认证 0.29 的差
需要每人 197 到 271 条。它买到的是粗大缺陷的筛查，以及核对 UTMOS 的排序和人是否一致。

**`listening_synthesiser` 量的是两个合成器。** edge-tts 3.871，voxcpm 3.476。
注意这个 3.871 和上面那个 4.023 是同一个 edge-tts 在两个包里的分——
两个包分开发、分开打分，引用时要说明是哪一包，不要混着比。

模型那边的完整数字和判定在
[`oscar0403/mindsurf-omni`](https://huggingface.co/oscar0403/mindsurf-omni)。

## 来源与许可

`listening_models` 和 `listening_synthesiser` 里是模型和合成器生成的音频。
`listening_emotion` 里是训练语料 [`gongjy/minimind_dataset`](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)
（在 ModelScope 上）助手一侧的真实音频，每条截到 10 秒上下。

**`listening_emotion` 不全是中文，发包的时候我们没有声明这件事。** 抽的 50 条里有 6 条是英语，
按 SenseVoice 的语种检测判的。没有粤语，但带口音的普通话排除不了——语种检测判不了口音。

整套材料按 CC-BY-NC-4.0 发布，不可商用。这条来自文本基座继承的语料许可，传导到每一个衍生物。
上游模型权重取自 [`jingyaogong/minimind-3o`](https://huggingface.co/jingyaogong/minimind-3o)。
