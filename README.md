# MindSurf Omni

语音 AI 产品的算法侧。输入是麦克风里的中文语音，输出是文本和语音回复。
模型内部把音频当作一种语言，音频 token 与文本 token 在同一条自回归序列里。

这是训练组的交付仓库。推理后端和前端需要的[接口契约](#4-接口契约)。

---

## 1. 交付产物

成品 `sft_merge_768.pth`，139,083,522 参数，我们自己的 Thinker 89.86M
加上游的 Talker 47.05M。

它不是重训出来的。父模型 `sft_graft_frozen` 上我们做过两次 DPO，
一次冲质量一次冲长度，两次都只动 Thinker，所以可以直接在权重上相加：

```
sft_merge = sft_graft_frozen + Δ(质量 DPO) + Δ(长度 DPO)
```

相加之前先量了两个 Δ 会不会互相抵消。91 个张量上余弦 +0.1329，近乎正交，
所以两个效应都能保住。语音那半（Talker 加 `audio_proj`，104 个张量）与父逐位相同。
一个配置跑完，没扫参数。

| | 地址 |
| --- | --- |
| 代码、文档、逐条证据 | 这个仓库 |
| 权重、tokenizer、配置 | [`oscar0403/mindsurf-omni`](https://huggingface.co/oscar0403/mindsurf-omni) |
| 人工盲听材料 | [`oscar0403/mindsurf-omni-listening`](https://huggingface.co/datasets/oscar0403/mindsurf-omni-listening) |

| | 状态 |
| --- | --- |
| 文本基座（89,864,448 参数，中文为主） | 已训练、已发布、已验证可接入 |
| 接口契约与桩服务 | 已冻结，后端可据此开工 |
| 原生音频路径（Thinker-Talker） | 已验收。我们的 Thinker 配上游的 Talker，重训中间那座桥 |
| 级联兜底路径（ASR → LLM → TTS） | 三段都接了：SenseVoice、Thinker（指 `MINDSURF_THINKER`）、合成器两选一（`MINDSURF_TTS=edge` 托管或 `voxcpm` 本地，默认不选） |

文本基座来自[上游预训练仓库](https://github.com/io-wy/MindSurf/tree/pretrain)，
权重在它的 Release 里。已实测该基座可直接加载进 MiniMind 的模型类，
最大绝对 logit 差 0.0；tokenizer 与 MiniMind / MiniMind-O 逐字节相同
（SHA-256 `71f32c68…`）。

---

## 2. 四条 KPI

延迟这条最好写。首音 145.1 ± 4.1 ms，P95 189，预算 3000，余量 16 倍。
噪声底 ±4.1 ms，这把尺子有门控资格，不是拍脑袋报的一个数。

规模 139M，线是 200M。KPI 写的是上限不是等号。

原生多模态那条要拆成三项说，因为三项的结果不一样。

- 打断：调用方说停就真停。取消到停 P50 0.26 ms、P95 0.40 ms，40 轮，预算 200 ms。
  我们特意把"停了"和"说自己停了"分开量：GPU 占用 0.5 秒内从 41% 掉到 0，
  同一轮完整跑完要 4953 ms，取消之后墙钟只有 132 ms。停的是计算本身。
- 音色克隆：seen 0.6487、unseen 0.6044，对父不劣化。更早那一轮对上游官方发布权重，
  12 个音色逐个比全赢。编解码器天花板是 0.8409，读这两个数要对着它读。
- 情绪：能力在，旋钮做不出来，见下面第二条边界。

产品可落地不归训练组，我们出的是接口契约、桩服务和推理参数。

### 人耳那一条

四个人盲听，每人一份随机顺序。我们的成品和上游官方发布权重打 2.738 对 2.738，
到小数点后三位相同，缺陷标记 43 对 42。参照上限 edge-tts 是 4.023。

在这之前，"与官方无法区分"这句话只有 UTMOS 一把自动尺子撑着。现在有一把
不可能继承 UTMOS 盲点的仪器给出了同一个判定。

产品上最痛的那一格也动了：中位回复从父模型的 26.0 秒降到 11.8 秒，
占用比从 13.7× 降到 6.2×，5 秒时还没说完的比例从 93% 降到 81%。空回复 0，循环 1。

---

## 3. 三条边界

这三条不是欠账。判据都写在跑之前，死因是量出来的。

**一、对改变长度的干预，"对话建模不退"这一轴在我们的预算内不可认证。**
四把尺子，四种死法。`chat_nll` 原版死在两个中立作者逐条同号只有 20.3%，
机制查出来是参考长度（r=+0.601）；按参考长度分层这条被证伪，控住之后同号率掉到 1.9%；
答得上率连一次已经认证过的塌方都读不出（+0.0872 落在 ±0.1061 里）；
长度配平版把原病灶修好了（同号 94.9%），却掉进镜像混淆，
分不开"建模更好"和"语域更配"。要闭合大概需要 3600 条探针，我们有 608，差六倍。
所以这不是"这一条没测"，是测了四次、每次都拿到了它为什么测不了。

选 `sft_merge` 就是选了这个代价。另一个候选 `sft_dpo2` 五条判据全过、没有星号，
它那一轴是认证过的，因为它没改长度。代价是中位回复仍然念 24.2 秒，
而 30 秒级的回复是我们唯一量出来的、会让人不想用第二次的缺陷。
两个都写进模型卡，只发一个的权重。

**二、情绪能力存在，但做不成可控旋钮。**
参考通路是真的：同一个说话人的两条不同情绪的参考喂进去，输出 F0 +54.7 Hz，
12 个音色全正，零训练。卡住的是条件化重训的前提。那个前提是标签可信，
而四个人的盲听把它证伪了：人对 `emotion2vec` 标签的一致率 31–42%，人对人 41–65%。
两个数都低说明任务本来就难；人彼此对得上、却都对不上标签，那是标签系统性偏离人耳。
条件化 A2A 我们也真跑过一轮，主判据 F0 零效应，而对照臂先过了，所以是真零不是读不出。
情绪和音色纠缠是学界的公开问题，解法要独立编码器加解耦损失再重训，
架构改动我们一开始就排除了。

**三、打断做到了执行，没做到判断。**
能停不等于会判断该停。后者要重叠音频加打断标注的语料，我们手上两份语料都没有，
合成的重叠只等于 VAD。端点决定现在在客户端。

这一页的数字全部取自
[`configs/release/headline_numbers.json`](configs/release/headline_numbers.json)，
那是对外数字的唯一真源，每条带 `source` 指向 [`artifacts/`](artifacts) 里产出它的逐条读数。
判定只有改善、劣化、无法区分三种；分辨不了我们关心的效应时标成仅报告，不参与判定。
阈值一律写在跑之前，中途没有改过。**怎么自己重算见 §6。**

---

## 4. 接口契约

不发明新协议，说 OpenAI 的。这样后端可以把已有客户端直接指过来，
也可以在我们的模型不稳时立刻指回托管服务做对照。这个退路比任何自研协议都值钱。

契约定义在 [`src/mindsurf_omni/contract.py`](src/mindsurf_omni/contract.py)，
它是唯一真源，文档与它冲突时以代码为准。

### HTTP

```
POST /v1/audio/transcriptions  语音转文本（Whisper API 兼容）
POST /v1/chat/completions  文本对话，支持 stream=true（SSE）
POST /v1/audio/speech  文本转语音
GET /v1/models  当前活跃路径、组件身份、许可
GET /v1/voices  可用音色
GET /v1/token-spec  特殊 token 规格（机器可读）
GET /v1/licence  完整许可链，含尚未核实的那几项
```

`GET /health` 报就绪度，逐部件。降级返回 200，全不可用才 503——
能转写、能出声的实例不该被摘出轮转。

### WebSocket

```
WS /v1/realtime
```

事件名沿用 OpenAI Realtime API 的子集，客户端不需要学新词汇：

| 上行 | 含义 |
| --- | --- |
| `input_audio_buffer.append` | 追加音频（base64 PCM16） |
| `input_audio_buffer.commit` | 说完了，开始回复 |
| `response.cancel` | 打断，立刻停止发声 |
| `session.update` | 改音色或情绪 |

| 下行 | 含义 |
| --- | --- |
| `response.text.delta` | 文本增量 |
| `response.audio.delta` | 音频增量（base64 PCM16），边生成边下发 |
| `response.audio.done` / `response.done` | 结束 |
| `error` | 出错，含可读原因 |

### 音频格式（写死，不协商）

| 方向 | 格式 |
| --- | --- |
| 上行 | PCM16 / 16 kHz / 单声道（SenseVoice 要 16k） |
| 下行 | PCM16 / 24 kHz / 单声道（Mimi 出 24k） |

重采样在服务端做。客户端不需要判断该用哪个采样率，需要判断就会有人判断错。

### 推理参数：文本能调，音频不能

| 轴 | 参数 | 值 | 谁定 |
| --- | --- | ---: | --- |
| 文本 | `temperature` | 0.7 | 你调（契约默认） |
| | `top_p` | 0.9 | 你调 |
| | `max_tokens` | 512 | 你调 |
| | 重复惩罚 | 1.0（关） | 写死，文本不加惩罚 |
| 音频 | temperature | 0.2 | 写死在 `stream_generate` 里 |
| | `top_k` | 50 | 写死 |
| | 重复惩罚 | 1.05 / 最近 3 码 | 写死 |

**`temperature` 和 `top_p` 碰不到音频。** 提高它期待"语音更有表现力"，
会得到完全一样的音频和更飘的文本。

音频那三个值扫过七组（在上游官方权重上，每组 24 条交替轮转），
没有一组打得过继承值，也没有一组被证明更差。
其中"去掉重复惩罚"那组有三条样本炸掉（逐例 CER 0.095 到 0.881、0.118 到 0.487、
0.067 到 0.420），失败模式是退化性重复——**那个重复惩罚不是装饰，别当死代码删掉**。
文本那三个值是契约默认，从没扫过，要调就调，但请自己测。

### 两条路径，同一个接口

```
                 ┌── native   Thinker-Talker 端到端，不经过文本
/v1/realtime ────┤
                 └── cascade  SenseVoice → Thinker → 合成器（edge / voxcpm）
```

切换是配置，不是改代码。调用方分辨不出是哪条在答，除非去问 `GET /v1/models`，
那里如实报告 `"path": "native" | "cascade"`。

两条都做是有意的。139M 规模能否把中文语音说好没有先例（MiniMind-O 自己也说中文
Talker 明显比英文难），而级联已被姊妹项目实测到端到端 P95 1.93 s。
产品不能只有一条能出声的路。

---

## 5. 许可（先看这一节）

| 资产 | 许可 | 可商用 | 核实过 |
| --- | --- | :-: | :-: |
| 文本基座 | CC-BY-NC-4.0（继承自训练数据） | 否 | 是 |
| MiniMind-O 代码 | Apache-2.0 | 是 | 是 |
| Mimi | CC-BY-4.0，要署名 | 是 | 是 |
| CAM++ | Apache-2.0 | 是 | 是 |
| 上游 Talker 权重 | 发布卡声明 apache-2.0 | 不明 | 否 |
| MiniMind-O 数据 | 卡上同时声明 apache-2.0 和 gpl-3.0，无逐文件映射 | 不明 | 否 |
| SenseVoice / emotion2vec | FunASR Model Open Source License Agreement v1.1 | 不明 | 否 |

权重继承数据的许可，且传导到所有微调结果。这条链上最严的是 CC-BY-NC，
所以当前产出物默认不可商用。`GET /v1/models` 会在响应里带上
`commercial_use_permitted: false`，让这件事不可能被忽略。

要做商业产品，得先换掉基座的训练数据。那是另一个项目，不是这一轮能解决的。

逐条依据、开放问题、以及要怎样才能改掉这个结论，写在
[`configs/release/licence.json`](configs/release/licence.json)。

---

## 6. 怎么自己重算我们报的数

上面每一个数字的逐条读数都在 [`artifacts/`](artifacts) 里，不用重跑训练、
也不用有卡就能核。`headline_numbers.json` 每一条的 `source` 指的就是那些文件。

### 语音：CER、UTMOS、静音率，以及对父模型的配对判定

```bash
python scripts/evaluate_speech.py   --candidate artifacts/merge/speech-sft_merge-mos.jsonl   --reference artifacts/merge/speech-parent-mos.jsonl
```

打印候选与参照各自的读数、噪声底、每个指标有没有门控资格，最后是逐样本配对的判定。
判官是 `paraformer-zh`，独立于被测模型——用模型自己的编码器给自己打分是循环论证，
共享的失败模式会互相抵消。

### 对话盲评：608 条探针

判过的每一对、判官身份、提示词 sha256、选边种子，全部落在
[`artifacts/blind608/`](artifacts/blind608)。`blind-merge-608.json` 是聚合结果
（608 对、169 平、赢 282，胜率 0.6424、噪声 0.0468、位置右侧占比 0.4465），
`blind-merge-608-pairs.json` 是逐对的原始判词。

重判要判官的 API key，会花钱：

```bash
JUDGE_API_KEY=... python scripts/blind_preference.py   --arm parent=artifacts/blind608/chat-sft_graft_frozen-608.json   --arm merge=artifacts/blind608/chat-sft_merge-608.json   --output <你的输出>.json
```

换个判官重判如果结论不同，那不代表旧数字错了，代表这个判定依赖判官——
那本身是关于仪器的发现，应该照实写进报告。

### 延迟与打断

`artifacts/latency-native-leak-fixed-2026-07-28.json` 是首音的逐轮读数，
[`artifacts/barge_in/`](artifacts/barge_in) 是打断的 40 轮读数
加上"计算真的停了"那份证明（墙钟对比与 GPU 占用采样）。
两件事分开量是有意的：只证明服务答得快，排除不了"先说停了、线程继续在卡上解完"。

### 人工盲听

四个人、三个包、十二份表，原件在
[`artifacts/listening_returned/`](artifacts/listening_returned)，
材料本身在 [`oscar0403/mindsurf-omni-listening`](https://huggingface.co/datasets/oscar0403/mindsurf-omni-listening)。

`scripts/listening_test.py score --pack <包>` 能算出 MOS 与一致率，
**但它要 `key.json`（揭盲表），而那个文件按纪律不入库**：
仓库是公开的、评分员也拿得到，答案入库就等于把答案发给了被测的人。
要复核这一项，找我们要那份表。

### 环境与自检

```bash
uv sync --extra dev --extra tts
.venv/Scripts/python -m pytest -q      # 631 passed / 7 skipped
python scripts/verify_delivery.py      # 交付齐备、文档与代码一致、数字的证据都在
```

别用 `uv run pytest`：这个项目的 dev 依赖是 extra 不是 group，`uv run` 不装它，
本机装了全局 pytest 时会静默拿全局解释器跑，那个环境没有 `soundfile`，
于是测的是另一个环境。

契约测试不依赖模型，先于模型存在。这正是重点：后端和客户端可以与模型并行开发，
契约里一个字段挪位置，两个组各赔一天。

---

## 7. 仓库里有什么

```
src/           推理服务与评测库
scripts/       训练、评测、复现用的命令行工具
tests/         631 项，契约与仪器的回归
configs/       探针集、语音系统提示、release/ 下的模型卡与对外数字真源
artifacts/     每个数字的逐条读数（音频与权重不入库）
assets/        tokenizer
examples/      一份能跑的客户端，抄走改
```

权重在 [`oscar0403/mindsurf-omni`](https://huggingface.co/oscar0403/mindsurf-omni)。

上游：[预训练仓库](https://github.com/io-wy/MindSurf/tree/pretrain)（基座怎么来的）、
[MiniMind-O](https://github.com/jingyaogong/minimind-o)（音频架构，Apache-2.0）、
[Mimi](https://huggingface.co/kyutai/mimi)（编解码器，CC-BY-4.0，钉了 revision）。
