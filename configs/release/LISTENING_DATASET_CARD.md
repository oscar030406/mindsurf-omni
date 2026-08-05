---
license: cc-by-nc-4.0
language:
  - zh
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

三组盲听材料，给三位评分员用。这里只有要听的音频和要填的表，没有答案。

## 你要下哪几个文件夹

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

## 三个人的编号是错开的

同一条音频在三个人的表里编号不同，这是故意的，所以不要互相对编号。
每个包里还混了重复条目，用来看同一个人前后给分是否一致。

## 答案表不在这里

生成这些包的脚本会同时写一份 `key.json`，记着每条音频出自哪个系统、
或者那条语料原本被标成什么情绪。包里的 `README.txt` 写着"不要打开 key.json"，
我们干脆没有传：这个仓库和我们的代码仓库里都没有它。
评分收齐之后，它会连同结果一起公开。

## 这批评分后来得出了什么

四个人填完了，三个包各自给出一条结论。写在这里是因为下载这份材料的人
应该先知道它被用来回答什么问题。

`listening_emotion` 那一包否掉了一个训练计划。人对 `emotion2vec` 自动标签的一致率
是 31–42%，而人对人是 41–65%。两个数都低说明任务本来就难；
人彼此对得上、却都对不上标签，那是标签系统性偏离人耳。
我们原本要拿这批标签做一轮情绪条件化重训，这个结果说明前提不成立，那一轮没有开。

`listening_models` 那一包给了这个项目唯一一把人耳尺子。我们的成品与上游官方发布权重
打 2.738 对 2.738，到小数点后三位相同，缺陷标记 43 对 42。
在这之前"与官方无法区分"只有 UTMOS 一把自动尺子撑着，
现在有一把不可能继承 UTMOS 盲点的仪器给出了同一个判定。
`listening_synthesiser` 那一包的参照上限 edge-tts 是 4.023，voxcpm 是 3.476。

模型那边的完整数字与判定在
[`oscar0403/mindsurf-omni`](https://huggingface.co/oscar0403/mindsurf-omni)。

## 来源与许可

`listening_models` 和 `listening_synthesiser` 里是模型和合成器生成的音频。
`listening_emotion` 里是训练语料 [`gongjy/minimind_dataset`](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)
（在 ModelScope 上）助手一侧的真实音频，每条截到 10 秒上下。

`listening_emotion` 不全是中文，发包的时候我们没有声明这件事。
抽的 50 条里有 6 条是英语，按 SenseVoice 的语种检测判的。没有粤语，
但带口音的普通话排除不了，语种检测判不了口音。这条补记于 2026-08-05，
评分那时已经收齐，所以它没有影响任何一位评分员填表。写在这里是因为
拿这份材料做二次分析的人需要知道。

整套材料按 CC-BY-NC-4.0 发布，不可商用。这条来自文本基座继承的语料许可，
传导到每一个衍生物。上游模型权重取自
[`jingyaogong/minimind-3o`](https://huggingface.co/jingyaogong/minimind-3o)，
其发布卡声明 apache-2.0，我们没有独立核实过这一条。
