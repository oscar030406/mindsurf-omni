# MindSurf Omni

语音 AI 产品的算法侧。输入是麦克风里的中文语音，输出是文本和语音回复。
模型内部把音频当作一种语言，音频 token 与文本 token 在同一条自回归序列里。

这是训练组的交付仓库。

## 最近修了什么（2026-08-10）

把整条链在本地跑起来之后（后端 + 这个仓库的引擎 + 成品权重 + SenseVoice + edge-tts），
量出来六个问题，都是我们这侧的，都已修并有回归测试：

| 修之前 | 修之后 |
| --- | --- |
| 服务只答当前这一句，「那明天呢」接不上上文 | 级联把历史带进 prompt，转写也回传给调用方（原生仍是单轮，[原因](docs/INTEGRATION.md)） |
| 按住说话没出声，它把静音听成韩文，再用英文答 9.6 秒 | 识别前先判有没有人声，没有就返回空转写 |
| 一个大请求能把整个服务冻住几分钟，其它会话全部收不到帧 | 识别挪到线程上跑，事件循环不再被占住 |
| `/health` 说 ready 时权重还在磁盘上，第一个用户等 32 秒 | 装配时就加载，第一个请求 713 ms |
| 文字要等语音合成的网络往返才下发，界面白等 1.6 秒 | 文字随生成即时下发 |
| 同一个问题问三遍三个样，`temperature` 沿用上游的 0.7 | 默认改 0.4，一致率 0.270 → 0.418（[数据](docs/INTEGRATION.md)） |

契约同时补齐了 `session.clear`，并删掉三个从来没有发出过的事件。

---

## 1. 交付物

成品 `sft_merge_768.pth`，139,083,522 参数（我们自己的 Thinker 89.86M
加上游的 Talker 47.05M，两半之和 136.92M，其余 2.17M 是把两半接起来的部分，
不计入任何一半）。它不是重训出来的，是在父模型上加两次 DPO 的权重差，
怎么来的见[训练说明 §2](docs/TRAINING.md)。

| | 地址 |
| --- | --- |
| 代码、文档、逐条证据 | 这个仓库 |
| 权重、tokenizer、配置 | [`oscar0403/mindsurf-omni`](https://huggingface.co/oscar0403/mindsurf-omni) |
| 人工盲听材料 | [`oscar0403/mindsurf-omni-listening`](https://huggingface.co/datasets/oscar0403/mindsurf-omni-listening) |

两条路径都能出声：原生（Thinker-Talker 端到端）和级联（SenseVoice → Thinker → 合成器）。
切换是配置，调用方分辨不出。接口契约已冻结。

---

## 2. 指标

| | 值 | 参照 |
| --- | ---: | --- |
| 参数量 | 139.08M | 上限 200M |
| 首个音频 | 145.1 ms（测量误差 ±4.1，P95 189） | 预算 3000 ms |
| 打断：取消到停 | 0.26 ms（P50）、0.40 ms（P95），40 轮 | 预算 200 ms |
| **人工盲听 MOS** | **2.738** | **上游官方发布权重 2.738** |
| 中位回复时长 | 11.8 秒 | 父模型 26.0 秒 |

**人工盲听的含金量最高。** 四个人各自拿一份随机顺序，我们的成品和上游官方权重
到小数点后三位相同，缺陷标记 43 对 42，参照上限 edge-tts 是 4.023。

音色克隆对父模型不退化；对上游官方权重 12 个音色逐个比全赢，
那一轮比的是嫁接体 `sft_graft`，不是成品。

**但 12 个里只有 6 个能投放**，名单和三档划分在[接入指南 §10.3](docs/INTEGRATION.md)。

数字全部取自
[`configs/release/headline_numbers.json`](configs/release/headline_numbers.json)，
每条带 `source` 指向 [`artifacts/`](artifacts) 里产出它的逐条读数。
判定只有三种：改善、劣化、无法区分；指标分辨不了我们关心的差别时标成「仅报告」，印出来但不参与通过或失败。

---

## 3. 边界

具体看[训练说明 §4](docs/TRAINING.md)。

1. **把回复变短之后，没法证明对话能力没跟着退。**
   不是退了，是测不出来——衡量对话能力的办法本身也在衡量长度。
   四种办法都试过，要测得准约需 3600 条测试提问，我们有608条。
2. **情绪能力存在，但做不到「只动情绪、不动身份」。**
   换一条带情绪的参考，输出就带情绪；但那条参考同时决定说话人是谁。
3. **打断做到了执行，没做到判断。**
   调用方说停就真停，但「什么时候该停」缺语料，两份训练语料都没有重叠音频。

---

## 4. 许可

**当前产出物不可商用。** 文本基座继承训练数据的 CC-BY-NC-4.0，
并传导到所有微调结果。`GET /v1/models` 会在响应里带上
`commercial_use_permitted: false`，让这件事不可能被忽略。

---

## 5. 怎么复测

逐条读数都在 [`artifacts/`](artifacts)，不用重跑训练。

```bash
uv sync --extra dev --extra tts

# 语音：CER、UTMOS、静音率，以及对父模型的逐条对照
python scripts/evaluate_speech.py \
  --candidate artifacts/merge/speech-sft_merge-mos.jsonl \
  --reference artifacts/merge/speech-parent-mos.jsonl

# 人工盲听（读 xlsx 要 openpyxl）
uv sync --extra listening
python scripts/listening_test.py score \
  --pack artifacts/listening_returned/listening_models \
  --key artifacts/listening_models/key.json

# 自检
.venv/Scripts/python -m pytest -q
python scripts/verify_delivery.py
```

盲听那条会打出两组数，因为有两种算平均的办法，**我们对外报的是第一种**：

| 办法 | 我们的成品 对 官方 |
| --- | --- |
| **逐条评分平均**（每一条打分各算一次） | **2.738 对 2.738** |
| 按片段均值再平均（每条片段各算一票） | 2.731 对 2.737 |

两个数不一样不是舍入。盲听包里故意混了重复片段，用来查评分员自己前后状况是否发生变化，
所以每条片段被打分的次数不等（有的五次有的四次）。
按片段那种算法会把听的人少的片段和听的人多的算成一样的加权。
**我们平均的是人的判断，不是片段的质量，所以每条打分各算一次。**
参照上限 edge-tts 是 4.023。

对话盲评要打分模型的 API key，命令和逐对判词在
[`artifacts/blind608/`](artifacts/blind608)。换一个打分模型重判如果结论不同，
那不代表旧数字错了，代表这个结论取决于用哪个模型来判。

**别用 `uv run pytest`**：dev 依赖是 extra 不是 group，`uv run` 不装它。

---

## 6. 仓库地图

```text
src/           推理服务与评测库
scripts/       训练、评测、复现用的命令行工具
tests/         664 项，契约与仪器的回归
configs/       测试提问集、语音系统提示、release/ 下的模型卡与数字来源
artifacts/     每个数字的逐条读数（音频与权重不入库）
docs/          两份正文：接入指南、训练说明
assets/        tokenizer
examples/      一份能跑的客户端
```

`configs/release/` 下那两份 `*_CARD.md` 是 Hugging Face 两个仓的 README。

上游：[预训练仓库](https://github.com/io-wy/MindSurf/tree/pretrain)（基座怎么来的）、
[MiniMind-O](https://github.com/jingyaogong/minimind-o)（音频架构，Apache-2.0）、
[Mimi](https://huggingface.co/kyutai/mimi)（编解码器，CC-BY-4.0，钉了 revision）。
