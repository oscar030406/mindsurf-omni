# 运行手册

出问题时按这个查。每条都写了怎么确认，不是「可能是……」。

---

## 1. 服务起不来

### 每个端点都返回 503

```json
{"detail": "no speech engine is configured; set MINDSURF_ENGINE to 'native' or 'cascade'"}
```

这是正常状态，不是故障。没配引擎时服务照常启动并如实说明，
这样后端能在模型就绪之前完成集成。配上就好：

```bash
MINDSURF_ENGINE=cascade docker compose up
```

### 启动直接退出

错误信息会点名缺的那一样东西：

```
the cascade path needs these, and they are not on disk:
tokenizer=/app/weights/tokenizer, codec=/app/weights/mimi
-- mount the weights directory or set the matching variable
```

按名字挂载或设对应变量。看到笼统的「engine unavailable」请报 bug，
那说明我们某处漏了具体信息。

### 起来了，但服务的是错的模型

**这个不会报错，只会让所有人对着一个旧模型讨论。** 成品是
`sft_merge_768.pth`，它前面还隔着 `sft_graft`、`sft_graft_frozen` 两代。
指错了一样能应答、`/health` 一样 ready。

```bash
curl -s localhost:8000/v1/models | python -m json.tool | grep -i checkpoint
```

对不上 [`headline_numbers.json`](../configs/release/headline_numbers.json) 里
`delivery.checkpoint` 写的名字就是指错了。服务器上的 `~/omni/serve_native.sh` 曾经
写死过被取代的那个，现在改了并在注释里写明了。

### `/health` 报 `degraded`，但每个组件都 `ready`

看 `not_ready` 那一栏。组件齐全不等于能应答——级联有三段，
任何一段没接上这个实例就答不完一轮。

`degraded` 是 200 不是 503：能转写、能出声的实例不该被摘出轮转。

### 聊天返回 503 说 "no text generator wired"

那一段还没接。`/v1/audio/transcriptions` 和 `/v1/audio/speech` 照常工作，
`/v1/chat/completions` 和 WebSocket 整轮对话要等 `MINDSURF_THINKER`。

### 出声那一段返回 503 说 "no synthesiser wired"

没设 `MINDSURF_TTS`。默认不选是有意的：一个合成器要连托管端点、另一个要吃显存，
两种代价都该是显式选择，不是继承来的默认值。

```bash
pip install -e ".[tts]"          # 托管：edge-tts，回复会离开本机
MINDSURF_TTS=edge MINDSURF_ENGINE=cascade docker compose up

pip install -e ".[tts-local]"    # 本地：VoxCPM-0.5B + torch，权重首个请求时加载
MINDSURF_TTS=voxcpm MINDSURF_ENGINE=cascade MINDSURF_DEVICE=cuda \
  python -m uvicorn mindsurf_omni.service.app:app --host 0.0.0.0 --port 8000
```

本地那个不在这个镜像里跑。镜像是 `python:3.12-slim` 加运行时依赖，
没有 torch、没有 CUDA，compose 也没要显卡，`MINDSURF_TTS=voxcpm` 在容器里会以
「装不了这个包」的 503 收场。它跑在有卡的宿主机上，和文本生成那一段一样。

接上之后 `GET /v1/models` 的 `components` 里会出现 `tts-edge` 或 `tts-voxcpm`。
**CER 测的是合成器有没有把回复说出来**，所以换合成器就是换被测对象，
报告里那个名字必须跟着换。

本地那个还有一条：**它不带情绪**。VoxCPM 没有 instruct 模式也没有韵律参数，
`emotion` 请求过去无处可放，而把指令拼进文本会被逐字念出来，
那正是 `instruction_leaked` 在防的事。edge 那条走的是 prosody，两者在这一项上不等价。

---

## 2. 音质问题

### 声音变调、像快放或慢放

采样率错配，不是模型问题。

```bash
curl -sI -X POST localhost:8000/v1/audio/speech -d '{"input":"测试"}' | grep -i x-sample-rate
```

响应头写的是多少就按多少播。硬编码 16000 去播 24000 的音频，就是这个症状。

### 每句之间音量跳变

服务端做了峰值归一，且只衰减不放大——放大近乎静音的片段会把底噪也放大成嘶声。
若仍跳变，检查客户端是否自己又做了一次增益。

### 句子之间有明显停顿或咔哒声

生成的片段首尾常带死气，服务端会裁掉并保留 50 ms 余量。
若客户端自己拼接 PCM，确认没有在片段之间插入静音帧。

### 换了音色但听起来没换人

**先查是不是那四个认不出的。** `cherry`、`ethan`、`chelsie`、`momo` 的
12 选 1 认人只有 ≤ 2/20，它们塌向 serena 和 moon。这不是 bug，是生成侧带的身份不够，
名单和三档划分在[能力边界 §2.4](CAPABILITIES.md)。**别把这四个当独立音色投放。**

### 模型把「用开心热情的语气说」念出来了

已知失败模式，不是随机的。情绪指令走独立字段，不要塞进 `input`：

```json
{"input": "今天天气真好", "emotion": "happy"}
```

字段用对了还出现，说明合成器漏读，这批音频要走一遍 `screen_batch` 复筛。
这个漏读是间歇性的，抽检会在它没搞砸的批次里找不到。

---

## 3. 延迟问题

### 首个音频来得太晚

先分账再优化。单一总时长只告诉你超了，分账才告诉你去修哪：

```bash
python scripts/smoke_service.py --base http://localhost:8000   # 打印首个增量耗时
```

最大的一项通常不是模型，是「什么时候允许开始合成」。等整段回复生成完再合成，
等于把整个生成时间都花在用户听到之前。级联在第一个句末标点就切；
原生每 4 个 Mimi 帧（约 320 ms）解一次。

### 首音忽然涨到秒级，而且越用越慢

**先看空闲时的 GPU 占用，这一条抓的是这个项目最贵的一个 bug。**

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 空闲时必须是 0 %
```

不是 0，就是又有生成在给已经走掉的调用方解码。引擎曾经把生成放在工作线程上
往无界队列灌，消费者走开之后生产者照样把整轮在 GPU 上解完，于是第 n 轮在和
n−1 个孤儿抢卡。它把首音从 145 ms 顶到 3356 ms，**三天里两次被误诊为「环境太吵」**。
已修（消费者退出时置 `stop` 事件，生产者每步检查），有回归测试。

### WebSocket 一直没有回复

先看是不是发了服务端不认识的事件——不认识的事件会收到 `error` 帧，不会静默。
收不到任何帧则是连接问题。

打断没生效：`response.cancel` 会取消正在跑的生成并清空缓冲区。
确认真停了要看两处，只看响应回得快是不够的：

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1   # 应在半秒内归零
```

---

## 4. 训练问题

### loss 看起来涨了

先别信眼睛。行组洗牌让每个位置看到不同数据，跨位置比较从一开始就被混淆。

```bash
python scripts/watch_training.py --log <日志> --metric audio
```

它按 step 对齐、算 bootstrap 噪声底，差异落在噪声内就报 `indistinguishable`
而不是命名方向。这条规则拦住的第一个错误结论是我们自己的：
把 epoch 3 的 step 7900 比 epoch 1 的 step 27500，看着像在发散，对齐后是平的。

### 日志半天不动，但进程还在

stdout 缓冲，不是卡住。上游 `Logger` 用 `print()` 不 flush，重定向到文件时按块落盘。
用磁盘上的硬证据确认：

```bash
ls -la --time-style=+%H:%M:%S <out>/sft_*.pth   # 每 1000 步写一次
ps -o etime=,time= -p <pid>                      # CPU 时间 ≈ 墙钟即在算
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

下次启动加 `PYTHONUNBUFFERED=1`。

### 训练被 OOM 杀掉

```
Killed          # 或 exit code 137
```

上游数据集把整张 parquet 读进内存，full 数据在 23 GB 机器上装不下。
parquet 元数据报的「未压缩」是行组内编码后的字节数，展开进 Arrow 要大得多。
走我们的启动器，它按行组懒读：

```bash
python scripts/train_omni.py --data_path ../dataset/sft_t2a.parquet ...
```

### 参数量报得比预期小

```
跳过shape不匹配的权重: {...}
Model Params: 113.13M       # 应该是 152.06M
```

**这个会静默毁掉整轮训练。** 训练脚本只暴露 `--hidden_size` 和 `--num_hidden_layers`，
其余走上游默认（GQA 4 头、FFN 1536），和我们的基座（MHA、FFN 3584）对不上，
`strict=False` 把 40 个张量默默跳过留在随机初始化。

我们的启动器把形状钉死并加了检查：Thinker 参数不等于 89,864,448 就拒绝启动。
看到这个报错说明没走启动器。

---

## 5. 评测问题

### 指标显示「仅报告」而不是有门控资格

这是设计如此。分辨不了目标效应的仪器可以报告、不能判定。
成品自己的单臂 CER 就是这样：它分辨得了 0.0529，而我们关心的效应是 0.0500，够不着，
所以有判定资格的是配对那两条。

要它有资格，得扩样本量或换仪器，不是放宽阈值。

### CER 高得离谱

先确认用的是第三方识别器，不是我们自己的音频编码器。
用模型的部件给模型打分是循环论证，共享的失败模式会互相抵消。
`require_independent_judge` 会拒绝服务组件，也会拒绝血缘相同的识别器。

### 对比结果说「无法区分」

差异落在合并噪声底的 3 倍以内。这不是工具太保守，是数据不支持任何方向。
要下结论就加样本量。

### 跑出来的数和文档里写的对不上

先看是不是口径不同而不是数不同。最容易撞的一处是人耳 MOS：
**对外报的 2.738 是逐条评分平均**，按片段均值再平均是 2.731 对 2.737，
差的不是舍入——重复片段让评分条数不均衡。`listening_test.py score` 两个都打并标了哪个是对外的。

```bash
python scripts/listening_test.py score \
  --pack artifacts/listening_returned/listening_models \
  --key artifacts/listening_models/key.json
```
