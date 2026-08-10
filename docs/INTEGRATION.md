# 接入指南

---

## 0. 接前须知

**一、有两条路可以出声，默认用级联。**
具体见 §5。原生的好处是快，用户说完 145 ms 就出第一声，级联那边整轮 P95 是 1.93 秒；
坏处是声音明显更糙：

| | 人工盲听（1–5 分） | 念错的字 |
| --- | ---: | ---: |
| 原生（模型自己发音） | 2.7 | 每 100 字错约 7 个 |
| 级联（交给 edge-tts 念） | 4.0 | 每 100 字错约 4 个 |

2.7 分大概是「听着费力，但能听懂」。这个差距不是没训好，
和上游官方权重跑出来的结果一样，差的是把声音压成 token 的那个编码器，
换个更好的模型也改善不了。

**二、12 个音色只上 6 个。**
拿模型生成的音频去让机器认「这是 12 个人里的谁」，20 句里能认对多少：

- **上这 6 个**：`arthur`、`serena`、`eric`、`uncle_fu`、`dylan`、`moon`，认对 19–20 句
- **别上这 4 个**：`cherry`、`ethan`、`chelsie`、`momo`，认对 2 句以下，
  实际听起来就是 serena 或 moon，用户切过去发现声音没变。
- **中间的 2 个**：`vivian` 16 句、`jennifer` 6 句，自己听后决定。

完整分档在 §10.3。

**三、情绪只能切换，不能像滑块那样调大调小。**
`emotion` 字段在级联那条路上正常用，有 `neutral`、`happy`、`care` 三档，
前端做成三个按钮没问题，做成连续滑块不行。

原生那条路连三档都给不了：能改情绪的那个输入同时也决定了「说话的是谁」，
调情绪就会连人一起换掉。细节见 §11 和[训练说明 §4](TRAINING.md)。

---

## 1. 快速启动

```bash
docker compose up
curl http://localhost:8000/v1/models
python examples/minimal_client.py
```

没配模型时每个端点返回 503，并在 `detail` 里说清缺的是什么。

要真的出声，起服务时选一条路（默认哪条都不选，见 §5），级联还要再选一个合成器：

```bash
MINDSURF_TTS=edge MINDSURF_ENGINE=cascade docker compose up
```

合成器有 `edge`（托管，回复会离开本机）和 `voxcpm`（本地，要显卡）两种，
都不选就只有转写和文本能用，出声那一段会返回 503。两者的差别和装法见 §13。

---

## 2. 协议：OpenAI 兼容

| 端点 | 用途 |
| --- | --- |
| `POST /v1/audio/transcriptions` | 语音转文本 |
| `POST /v1/chat/completions` | 文本对话，`stream=true` 走 SSE |
| `POST /v1/audio/speech` | 文本转语音 |
| `GET /v1/models` | 当前路径、组件身份、许可 |
| `GET /v1/voices` | 可用音色 |
| `GET /v1/token-spec` | 特殊 token 规格（机器可读） |
| `GET /v1/licence` | 完整许可链 |
| `WS /v1/realtime` | 端到端流式语音 |
| `GET /health` | 就绪度，逐部件；降级返回 200，全不可用才 503 |

## 3. 音频格式是固定的

| 方向 | 格式 |
| --- | --- |
| 上行 | PCM16 / 16 kHz / 单声道 |
| 下行 | PCM16 / 24 kHz / 单声道 |

`POST /v1/audio/speech` 的响应头里带 `X-Sample-Rate` 和 `X-Encoding`，别在客户端写死。
写死的后果是换个合成器就整体变调，见 §14 第一条。

---

## 4. WebSocket 实时链路

事件名沿用 OpenAI Realtime API 的子集。

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/realtime");

// 连上先收 session.created，里面有采样率，按它走
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  switch (message.type) {
    case "response.text.delta":  appendText(message.delta); break;
    case "response.audio.delta": playPcm(atob(message.audio)); break;
    case "response.done":        turnFinished(); break;
    case "error":                console.error(message.error.message); break;
  }
};

// 边说边发
ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: base64Pcm }));
// 说完
ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
// 打断
ws.send(JSON.stringify({ type: "response.cancel" }));
```

音频是边生成边下发的，不是等整段做完。收到第一个 `response.audio.delta` 就可以开始播。
这正是延迟预算里最大的一项，等整段会把整个生成时间都花在用户听到之前。

不认识的事件会收到 `error` 而不是被忽略。

### 上下文会被裁剪但显式

`response.done` 带 `context`：

```json
{ "type": "response.done", "context": { "turns": 4, "used_tokens": 820, "budget": 2560, "dropped_turns": 2 } }
```

四个字段分别是：现在上下文里还剩几轮（`turns`）、用掉多少 token（`used_tokens`）、
上限多少（`budget`）、这次为了腾地方扔掉了几轮（`dropped_turns`）。

为什么会扔：语音比文字占地方得多。一秒音频等于 12.5 个 token（Mimi 编码器的帧率），
十秒的一句话就是 125 个，同样内容打成字只要十几个。聊几轮就会超过 `budget`，
服务端只能丢历史，从最早的一轮开始丢。

所以 `dropped_turns` 大于 0 的时候请记一条日志，或者在界面上提示一下。
不然用户会看到「刚说过的事它不记得了」，现象和 bug 一模一样，但其实什么都没坏。

换一个人用的时候发一条 `session.clear`：

```javascript
ws.send(JSON.stringify({ type: "session.clear" }));
```

不发的话，上一个人说过的内容会留在下一个人的上下文里。

目前只测到三轮，三轮之内不崩，也不需要去压缩历史。
逐轮的测试结果在 §9.3，十轮以上没测过。

---

## 5. 两条路径

```text
                 ┌── native   端到端，音频 token 直接过模型，不经过文本
/v1/realtime ────┤
                 └── cascade  语音识别 → 文本 → 语音合成
```

走哪条由服务端配置决定，客户端代码不用改，请求和事件完全一样。
想知道现在跑的是哪条，看 `GET /v1/models` 里的 `"path": "native" | "cascade"`。

为什么留两条：139M 这么小的模型能不能把中文说清楚，之前没有先例，
所以级联是保底的那条，它已经实测到整轮 P95 1.93 秒。

---

## 6. 情绪与音色怎么传

情绪走独立字段，不要塞进文本：

```json
{ "input": "今天天气真好", "emotion": "happy" }
```

塞进文本的后果是模型可能把指令念出来。

音色不是训出来的。常规做法是拿某个人的录音再训一遍、得到一套新权重，该项目不这么做：
权重一个字节都不改，生成前把一段该音色的参考音频放进上下文，模型照着它的嗓音往下说。

参考音频不是 wav 直接喂进去，预处理成两样东西：

| 字段 | 是什么 |
| --- | --- |
| `ref_codes` | 参考音频过 Mimi 编码器得到的码带 |
| `spk_emb` | 192 维说话人向量（CAM++ 抽的），嗓音特征的压缩表示 |

两样都存在 `model/speaker/voices.pt` 里，12 个音色现成，不用生成。
想加第 13 个人：录 5 到 10 秒，抽出这两样存进音色包就行。

接线上做法：前端选了 `arthur`，服务端从音色包里取出 `arthur` 对应的 `ref_codes` 和
`spk_emb`，作为参数传给推理。
**这一段现在还没接**：服务端不认音色名，`GET /v1/voices` 也只返回 `default` 一条。

---

## 7. 推理参数：文本能调，音频不能

### 7.1 先确认加载的是哪套权重

级联的 `MINDSURF_THINKER` 和原生路径用的是同一个 checkpoint。该指向哪个，
以 [`headline_numbers.json`](../configs/release/headline_numbers.json) 里的
`delivery.checkpoint` 为准，现在是 `sft_merge_768.pth`。

**指错了不会报错。** 服务照样起，`/health` 照样 ready，从外面完全看不出来，只是答得更差。

原生路径中负责发声的那半（Talker）必须是同一个 checkpoint 里配套的，不能拿别的凑。
拿只训过文字的 Talker 去克隆音色，240 条里只能成功 3 条。加载器会检查，
设错会直接拒绝加载，不会静默降级。

### 7.2 `temperature` 和 `top_p` 只作用于文本

它们碰不到音频。想让声音更有表现力而去调高 temperature，
结果是音频一模一样，文本反而更飘。声音的部分不受请求参数控制，见下表右侧。

| 轴 | 参数 | 值 | 状况 |
| --- | --- | ---: | --- |
| 文本 | `temperature` | 0.7 | 可调（契约默认，`contract.py`） |
| | `top_p` | 0.9 | 可调 |
| | `max_tokens` | 512 | 可调 |
| | 重复惩罚 | 1.0（关） | 写死，`native.py`，文本不加惩罚 |
| 音频 | temperature | 0.2 | 写死在上游 `stream_generate` 里 |
| | `top_k` | 50 | 写死 |
| | 重复惩罚 | 1.05 / 最近 3 码 | 写死 |

### 7.3 音频的参数不要动

试过七组不同的取值（每组 24 条，和现值交替轮转着测），大同小异。

**那个 1.05 的重复惩罚不是装饰，别当死代码删掉。** 关掉它，24 条里有 3 条彻底炸掉：
错字率从 0.095 涨到 0.881、0.118 涨到 0.487、0.067 涨到 0.420，
表现是同一个音反复念个不停。平均值看不出区别，是因为少数几条灾难样本把波动撑得太大，掩盖了它。

### 7.4 文本的参数没测过，可以随意调动

文本这边的 `temperature 0.7 / top_p 0.9` 是上游的默认值，未测试过。
调了之后把改动前后的效果测出来告诉我，这边没有这一轴的读数。

## 8. 出错时

| 状态 | 含义 | 该做什么 |
| --- | --- | --- |
| 503 | 没配引擎，或那一段还没接上 | 看 detail，它会点名缺的是哪个变量、哪一段 |
| 400 | 请求体没有音频 | 检查是否真的发了 PCM |
| WS `error` 帧 | 事件类型不认识，或缓冲区为空 | 看 `error.message` |

错误信息会写清楚是什么坏了、怎么修。
碰到看不懂的错误信息报给我谢谢喵。

---

## 第二部分：能力边界

打断能停下来，但「什么时候该停」要其他组判断；
音色克隆的质量不输上游，但 12 个音色只有 6 个听得出是不同的人；
情绪能切换，但不能做成滑块式连续变量来滑调。

---

## 9. 语音打断

### 9.1 停得下来

发 `response.cancel`，服务端不只是不再往下发音频，是把那次生成整个掐掉，
GPU 也跟着停。实测（成品权重，原生路径，40 轮）：

| 量 | 要求 | 实测 |
| --- | --- | --- |
| 从发出 cancel 到收到 `response.done(cancelled)` | ≤ 200 ms | P50 0.26 ms / P95 0.40 ms |
| 收到 done 之后还会漏过来几帧音频 | 只记录 | 中位 0，最多 1 |

```text
一个完整回合        4953 ms
被打断的回合         132 ms      只花了完整回合的 2.7%
打断瞬间 GPU  [41, 27, 27] %    0.5 秒后  [0, 0, 0] %
```

所以打断之后不用等，也不用担心后台还在偷偷算。最多有 1 帧音频已经在路上，客户端丢掉即可。

### 9.2 「什么时候该停」没做

服务端只能做到「你叫停就停」。用户是真想插话还是只是「嗯嗯」附和，这个判断在客户端，
由其他组决定发不发 `response.cancel`。为什么没做见[训练说明 §4](TRAINING.md) 第三条边界。

### 9.3 多轮检测：三轮之内不崩，上下文不是瓶颈

50 组对话 × 3 轮，第 2、3 轮全部是省略句。

| 轮 | 用上历史 | 答得上 | 空 | 循环 |
| --- | ---: | ---: | ---: | ---: |
| 1 | | 42% | 0 | 0 |
| 2 | 82% | 24% | 0 | 0 |
| 3 | 82% | 14% | 0 | 0 |

三轮之内不会塌，也不需要自己压缩历史，上下文利用率不随深度衰减。
答得上的比例随深度掉，但第 2、3 轮的问题是刻意设计的省略句，题目本身在变难，
这个下降里有多少来自上下文、多少来自题目，没有拆开分析。
十轮以上没测，那时上下文长度才可能变成问题。

## 10. 音色克隆

怎么传见 §6，这一节讲的是效果到什么程度、哪几个能投放。
12 个现成音色在 `model/speaker/`，每个的参考录音 5.1 到 9.4 秒。

### 10.1 怎么解读相似度分数

常见的读法是「相似度 0.65，离 1.0 还差得远」。这不对，因为 1.0 根本达不到。

把存好的参考音频原样解码回来，再重新抽一次说话人向量，和存档的相比，
相似度也只有 0.8409（12 个音色在 0.7888 到 0.8964 之间）。中间差掉的部分是编码器
（Mimi，8 码本、每秒 12.5 帧）压缩时丢掉的身份信息。
所以上游报的 0.6472（训练见过的音色）和 0.5654（没见过的），
应该读成「拿到了编码器允许范围的 76% 和 68%」。

### 10.2 与参考录音的相似度

12 个音色各念 20 句，和父模型逐条对照（同一句两边各念一遍再相减）：
见过的音色 **0.6487** 对 0.6470，没见过的 **0.6044** 对 0.6054。
差别在测量误差 ±0.015 以内，也就是说换成我们的权重，音色相似度没有退化。

### 10.3 与参考身份的相似度

10.2量的是「像不像那条参考录音」，10.3回答「听起来还是不是这个人」。
后者要这么测：拿每条生成音频去和 12 个音色的向量逐一相比
下表 rank-1 那一列就是 20 句里认对了几句。

下面这一列是在`sft_merge`上测量的，20 句固定文本、seed 7：

| 档 | 音色 | rank-1 | 结论 |
| --- | --- | ---: | --- |
| 认得出 | serena、eric、uncle_fu、dylan、arthur | 20/20 | 换到这几个，听感上是换了人 |
| 认得出 | moon | 17/20 | 同上，但余量不大 |
| **边缘** | vivian | 16/20 | 刚过线 |
| **边缘** | jennifer | 6/20 | 感觉不太行 |
| **认不出** | cherry、ethan、momo | 0/20 | **别当独立音色投放**，它们塌向 serena 和 moon |
| **认不出** | chelsie | 1/20 | 同上 |

**这张表只能分档，不能排名。** 一共才 20 句，上下差 3 条属于正常抖动，
所以 20/20、17/20、16/20 之间不要读出高低。

那四个认不出的不是测法的问题：把参考录音原样解码回去认，12 个全认对。
**是模型生成的音频里带的身份信息不够。**

## 11. 情绪控制

**换一条带情绪的参考录音，输出就带情绪，这做得到。**
同一个人的平静版和激动版两条参考，输出音高差 +54.7 Hz，12 个音色全是这个方向，不用训练。

**但它做不成一个旋钮。** 原生路径上唯一能影响情绪的输入就是说话人向量，
而这个向量同时就是「这个人是谁」，改变它必然连身份一起改变。
换句话说，**你没法在不换人的前提下把「开心」调大一点**。
具体训练尝试见[训练说明 §4](TRAINING.md) 第二条边界。

### 11.1 情绪接法

**要情绪滑块就挂在级联上。** 级联的 `EDGE_PROSODY` / `EMOTION_INSTRUCTIONS` 三档照常用。

**原生路径只能换参考。** 想要某个音色的开心版，就自备一条这个人开心时说话的**真实录音**，
当成一个新音色走 §6 那条通路加进去，不需要新接口。
但它是「换了一条参考」，不是「把情绪调大了」。

**情绪一定走 `emotion` 字段，不要拼进 `input`。**

---

## 12. 换 checkpoint 不用改配置

这个 checkpoint 的两半（负责思考的和负责发声的）大小不一样，
所以加载器是从权重文件里读出形状的，不看配置。
读到不认识的形状它会拒绝加载。
详细见[训练说明 §1](TRAINING.md)。

---

## 第三部分：问题排查

## 13. 服务起不来

### 每个端点都返回 503：正常状态

```json
{"detail": "no speech engine is configured; set MINDSURF_ENGINE to 'native' or 'cascade'"}
```

```bash
MINDSURF_ENGINE=cascade docker compose up
```

### 启动直接退出

```text
the cascade path needs these, and they are not on disk:
tokenizer=/app/weights/tokenizer, codec=/app/weights/mimi
-- mount the weights directory or set the matching variable
```

按名字挂载或设对应变量。看到笼统的「engine unavailable」请报给我谢谢喵。

### 起来了，但服务的是错的模型

成品是`sft_merge_768.pth`，它前面还隔着 `sft_graft`、`sft_graft_frozen` 两代。
指错了一样能应答、`/health`一样能起。

```bash
curl -s localhost:8000/v1/models | python -m json.tool | grep -i checkpoint
```

对不上 [`headline_numbers.json`](../configs/release/headline_numbers.json) 里
`delivery.checkpoint` 写的名字就是指错了。

### `/health` 报 `degraded`，但每个组件都 `ready`

看 `not_ready` 那一栏，它列的是还缺哪一段。级联要转写、生成、合成三段齐了才答得完一轮，
组件本身没坏不代表走得通。

### 聊天返回 503 说 "no text generator wired"

那一段还没接好。`/v1/audio/transcriptions` 和 `/v1/audio/speech` 照常工作，
`/v1/chat/completions` 和 WebSocket 整轮对话要等 `MINDSURF_THINKER`。

### 出声那一段返回 503 说 "no synthesiser wired"

没设 `MINDSURF_TTS`，即没告诉服务用哪个合成器把文字念出来。
两个都不默认选，是因为一个要把文字发到外网，一个要吃显卡。

**edge：托管的，回复文字会发到微软的服务，本机不用显卡。**

```bash
pip install -e ".[tts]"
MINDSURF_TTS=edge MINDSURF_ENGINE=cascade docker compose up
```

**voxcpm：本地的，不出网，但要显卡，而且不能在 docker 里跑。**

```bash
pip install -e ".[tts-local]"
MINDSURF_TTS=voxcpm MINDSURF_ENGINE=cascade MINDSURF_DEVICE=cuda \
  python -m uvicorn mindsurf_omni.service.app:app --host 0.0.0.0 --port 8000
```

镜像里没装 torch、没有 CUDA，compose 也没申请显卡，所以在容器里设 `voxcpm` 只会得到
一个「装不了这个包」的 503。它得在有卡的机器上起。

确认接上：`GET /v1/models` 的 `components` 里会多出 `tts-edge` 或 `tts-voxcpm`。

### voxcpm 没有情绪，切过去情绪滑块会失效

edge 的情绪是靠语速和音高实现的：服务端把 `emotion` 翻译成 `rate` / `pitch` 两个参数交给它。
VoxCPM 既不接受「用开心的语气说」这种文字指令，也没有语速音高参数，
`emotion` 传过去无处可放，只能被忽略。想绕过去把指令拼进正文，它会一字不落地念出来（见 §14）。

---

## 14. 音质问题

### 声音变调、像快放或慢放

采样率错配，不是模型问题。

```bash
curl -sI -X POST localhost:8000/v1/audio/speech -d '{"input":"测试"}' | grep -i x-sample-rate
```

响应头写的是多少就按多少播。

### 每句之间音量忽大忽小

服务端已经做了音量归一，只压不抬——把近乎静音的片段抬上来，会连底噪一起抬成嘶嘶声。
还跳的话，检查客户端是不是自己又做了一次增益。

### 句子之间有明显停顿或咔哒声

生成的片段首尾常带一小段杂音，服务端会裁掉，两头各留 50 ms 余量。
如果是客户端自己拼 PCM，确认没有在片段之间插入静音帧。

### 换了音色但听起来没换人

先查是不是那四个认不出的。`cherry`、`ethan`、`chelsie`、`momo` 的
12 选 1 认人只有 ≤ 2/20，它们塌向 serena 和 moon。不是 bug，是生成的音频带的身份不够，
完整名单和分档在 §10.3。**别把这四个当独立音色投放。**

### 音频里念出了「用开心热情的语气说」

已知的失败模式，不是随机的。情绪指令走独立字段，不要塞进 `input`：

```json
{"input": "今天天气真好", "emotion": "happy"}
```

字段用对了还念出来，说明是合成器那边漏读，这批音频要走一遍 `screen_batch` 复筛。

---

## 15. 延迟问题

### 首个音频来得太晚

先分账再优化。单一总时长只告诉你超时了：

```bash
python scripts/smoke_service.py --base http://localhost:8000   # 打印首个增量耗时
```

大头通常不是模型慢，是「等到什么时候才开始出声」。等整段回复写完再去合成，
等于把整个生成时间都花在用户听到第一个字之前。我们的做法是：
级联在第一个句号问号处就切一段去合成；原生每攒够 4 帧（约 320 ms）就解码下发一次。

### 首音忽然涨到秒级，而且越用越慢

先看空闲时的 GPU 占用。

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 空闲时必须是 0 %
```

不是 0，说明还有生成在给已经断开的调用方跑，占着卡，后来的请求就都慢了。
这个问题已修（客户端退出时置 `stop` 事件，服务端每步检查），有回归测试盯着。

### WebSocket 一直没有回复

先看是不是发了服务端不认识的事件——不认识的事件会收到 `error` 帧，不会静默。
收不到任何帧则是连接问题。

打断没生效：`response.cancel` 会取消正在跑的生成并清空缓冲区。

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1   # 应在半秒内归零
```

---

## 16. 评测问题

### 指标标着「仅报告」

意思是这个指标这次**不参与通过还是失败的判定，只印出来参考**。

原因是它分辨不了我们关心的那点差别。举例：单看一边的错字率，这批数据能分辨的最小差别是
0.0529，而我们关心的差别是 0.0500，比它的分辨能力还小，所以它给出什么结论都不作数。
**要让它作数得加样本量或者换个测法，不是把阈值放宽。**

### 错字率（CER）高得离谱

先确认用来听的是第三方识别器，不是我们自己的音频编码器。
拿模型自己的部件给模型打分是循环论证，`require_independent_judge` 会拒绝服务组件，
也会拒绝和被测对象同源的识别器。

### 对比结果写着「无法区分」

意思是两边的差别落在这批数据的测量误差里，**数据不支持说谁好谁坏**。
想下结论就得加样本量（虽然加了也可能还是落在噪声内）。
