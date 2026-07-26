# MindSurf Omni

语音 AI 产品的算法侧。输入是麦克风里的中文语音，输出是文本和语音回复；模型内部把
音频当作一种语言，音频 token 与文本 token 在同一条自回归序列里。

**这是训练组的交付仓库。** 推理后端和前端不需要读这里的训练代码，只需要
[接口契约](#接口契约)。

---

## 1. 现在能用什么

| | 状态 |
| --- | --- |
| 文本基座（89,864,448 参数，中文为主） | ✅ 已训练、已发布、已验证可接入 |
| 接口契约与桩服务 | ✅ 已冻结，后端可据此开工 |
| 原生音频路径（Thinker-Talker） | 🔨 训练中（第三轮／嫁接体）。**第一轮那个「配方缺陷」2026-07-26 查出并不存在**——上游 `train.sh` 写的就是那套参数，见 [配方 bug 不是 bug](docs/experiments/2026-07-26-recipe-bug-was-not-a-bug.md) |
| 级联兜底路径（ASR → LLM → TTS） | ✅ 三段都接了：SenseVoice、Thinker（指 `MINDSURF_THINKER`）、合成器两选一（`MINDSURF_TTS=edge` 托管 / `voxcpm` 本地，**默认不选**） |

文本基座来自[上游预训练仓库](https://github.com/io-wy/MindSurf/tree/pretrain)，
权重在它的 Release 里。已实测该基座可直接加载进 MiniMind 的模型类，
**最大绝对 logit 差 0.0**；tokenizer 与 MiniMind / MiniMind-O 逐字节相同
（SHA-256 `71f32c68…`）。

---

## 2. 接口契约

**不发明新协议，说 OpenAI 的。** 这样后端可以把已有客户端直接指过来，也可以在我们的
模型不稳时立刻指回托管服务做对照——这个退路比任何自研协议都值钱。

契约定义在 [`src/mindsurf_omni/contract.py`](src/mindsurf_omni/contract.py)，
它是唯一真源，文档与它冲突时以代码为准。

### HTTP

```
POST /v1/audio/transcriptions   语音转文本（Whisper API 兼容）
POST /v1/chat/completions       文本对话，支持 stream=true（SSE）
POST /v1/audio/speech           文本转语音
GET  /v1/models                 当前活跃路径、组件身份、许可
GET  /v1/voices                 可用音色
GET  /v1/token-spec             特殊 token 规格（机器可读）
```

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
| `session.update` | 改音色/情绪 |

| 下行 | 含义 |
| --- | --- |
| `response.text.delta` | 文本增量 |
| `response.audio.delta` | 音频增量（base64 PCM16），**边生成边下发** |
| `response.audio.done` / `response.done` | 结束 |
| `error` | 出错，含可读原因 |

### 音频格式（写死，不协商）

| 方向 | 格式 |
| --- | --- |
| 上行 | PCM16 / **16 kHz** / 单声道（SenseVoice 要 16k） |
| 下行 | PCM16 / **24 kHz** / 单声道（Mimi 出 24k） |

重采样在服务端做。客户端不需要判断该用哪个采样率——需要判断就会有人判断错。

### 两条路径，同一个接口

```
                 ┌── native   Thinker-Talker 端到端，不经过文本
/v1/realtime ────┤
                 └── cascade  SenseVoice → Thinker → CosyVoice2
```

切换是配置，不是改代码。调用方分辨不出是哪条在答，除非去问 `GET /v1/models`——
那里如实报告 `"path": "native" | "cascade"`。

**两条都做是有意的**：140M 规模能否把中文语音说好没有先例（MiniMind-O 自己说中文
Talker 明显比英文难），而级联已被姊妹项目实测到端到端 P95 1.93 s。产品不能只有一条
能出声的路。

---

## 3. 许可（先看这一节）

| 资产 | 许可 | 可商用 |
| --- | --- | :-: |
| 文本基座 | **CC-BY-NC-4.0**（继承自训练数据） | ❌ |
| MiniMind-O 代码 | Apache-2.0 | ✅ |
| MiniMind-O 数据 | 未声明，上游含 VoiceAssistant-400K 等 | ❓ 待查 |
| SenseVoice / Mimi / CAM++ | 各自许可 | ❓ 待查 |

**权重继承数据的许可，且传导到所有微调结果。** 这条链上最严的是 CC-BY-NC，
所以**当前产出物默认不可商用**。`GET /v1/models` 会在响应里如实带上
`commercial_use_permitted: false`，让这件事不可能被忽略。

要做商业产品，得先换掉基座的训练数据——那是另一个项目，不是这一轮能解决的。

---

## 4. 开发

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run mypy src tests scripts
```

契约测试不依赖模型，先于模型存在——这正是重点：后端和客户端与模型并行开发，
契约里一个字段挪位置，两个组各赔一天。

---

## 5. 相关文档

- **[交接](docs/HANDOVER.md)：接手先读这份**
- [架构](docs/ARCHITECTURE.md)：数据怎么流，以及五个不显然的决定
- [接入指南](docs/INTEGRATION.md)：后端与前端要的全部，不必读训练代码
- [运行手册](docs/RUNBOOK.md)：出问题时按症状查，每条都写了怎么确认
- [评测](docs/EVALUATION.md)：怎么测，以及测出来的东西能说什么、不能说什么
- [决策记录](docs/DECISIONS.md)：八个有分歧的决定，各自的依据与推翻条件
- [行动指南](docs/ACTION_PLAN.md)：七天计划、跳出上游框架的六处决定及其理由、风险与退路
- [基座接入](docs/CONVERSION.md)：文本基座怎么变成 Thinker，以及等价性怎么验
- [吞吐](docs/THROUGHPUT.md)：GPU 到底在不在等 CPU（实测推翻了直觉）
- [上游预训练仓库](https://github.com/io-wy/MindSurf/tree/pretrain)：基座怎么来的
- [MiniMind-O](https://github.com/jingyaogong/minimind-o)：音频架构的上游（Apache-2.0）
