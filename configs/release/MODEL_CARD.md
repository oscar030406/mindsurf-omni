---
license: cc-by-nc-4.0
language:
  - zh
pipeline_tag: any-to-any
tags:
  - speech-to-speech
  - spoken-dialogue
  - chinese
---

# mindsurf-omni（`sft_merge`）

139M 参数的中文语音对话模型。语音进、语音出，文本与音频 token 在同一条自回归序列里。

这张卡只写量过的数字。每一条都带噪声底与判定，判定只有改善、劣化、无法区分三种；
分辨不了我们关心的效应时标成仅报告，不参与判定。所有数字取自代码仓库的
[`configs/release/headline_numbers.json`](https://github.com/oscar030406/mindsurf-omni/blob/main/configs/release/headline_numbers.json)，
那是对外数字的唯一真源，每条带 `source` 指向产出它的证据文件。

发布的权重是 `sft_merge_768.pth`，
sha256 `d67c744019f786e851247fea712953f62573161b08318f704a872bee89a82753`。

## 它是怎么来的

不是重训出来的。父模型 `sft_graft_frozen` 上做过两次目标不同的 DPO，
一次冲质量一次冲长度，两次都只动 Thinker，所以可以直接在权重上相加：

sft_merge = sft_graft_frozen + Δ(质量 DPO) + Δ(长度 DPO)

相加之前先量了两个 Δ 会不会互相抵消。91 个张量、9478 万参数上余弦 +0.1329，
近乎正交，所以两个效应都能保住。合并体位移 |Δ| = 1.7545，两者分别是 0.9617 与 1.3499，
完全正交时应为 1.6575。语音那半（Talker 加 `audio_proj`，104 个张量）与父逐位相同。

一个配置（α=β=1）跑完，没扫参数。

## 结构

| | 参数量 | 来源 |
| --- | ---: | --- |
| Thinker | 89,864,448 | 我们自己的中文基座 |
| Talker | 47,050,754 | 上游 MiniMind-o 发布权重 |
| 合计 | 139,083,522 | |

Talker 只通过 `talker.embed_proj` 读 Thinker 的隐状态。接口窄到可以换半边，
这个模型换的就是那半边。音频编解码用 [Mimi](https://huggingface.co/kyutai/mimi)，
8 码本、12.5 Hz。

## 量过的

### 语音

对父模型，固定 160 句 teacher-forcing、逐样本配对，判官 `paraformer-zh`（独立于被测模型）：

| 轴 | 本模型 | 判定 |
| --- | ---: | --- |
| CER | 0.0962 ± 0.0176 | `indistinguishable (+0.0059, ±0.0314)` |
| UTMOS | 2.4399 ± 0.0498 | `indistinguishable (-0.0127, ±0.0985)` |
| 静音率 | 0/160 | 同 |
| 音色克隆余弦（12 音色 × 20 句） | seen 0.6487 / unseen 0.6044 | 对父 0.6470 / 0.6054，底 ±0.015，不劣化 |

单臂 CER 只能仅报告：它分辨得了 0.0529，而我们关心的效应是 0.0500，够不着。
有判定资格的是配对那两条。克隆的编解码天花板是 0.8409，
上游报的 seen 0.6472 应理解成天花板的 76%。

**0.0962 不要拿去和级联的 0.0359 直接比。** 160 条参考文本里 71 条含阿拉伯数字，
它们的 CER 是 0.1482，其余 89 条是 0.0548。差的大半不是发音，
是「2021 念成二零二一」这种正确行为被判官罚，**而级联那条也被同样罚**，
所以两个数在这一项上不可比。折叠数字口径下全部是 **0.0730**。

这三行可以自己重算，仓库里有逐样本读数：

```bash
python scripts/evaluate_speech.py \
  --candidate artifacts/merge/speech-sft_merge-mos.jsonl \
  --reference artifacts/merge/speech-parent-mos.jsonl
```

### 人耳

四位评分员、盲听、每人一份随机顺序。这是这个项目唯一一把人耳尺子。

| 系统 | MOS | 缺陷标记 |
| --- | ---: | ---: |
| edge-tts（参照上限） | 4.023 | 0 |
| 我们的 graft 臂 | 2.738 ± 0.120 | 43 |
| 上游官方 `sft_omni` | 2.738 ± 0.120 | 42 |
| voxcpm | 3.476 | |

在这之前，"与官方无法区分"这句话只有 UTMOS 一把自动尺子撑着。现在有一把
不可能继承 UTMOS 盲点的仪器给出了同一个判定。

这句话的读法要精确。盲听包里的音频出自 graft 臂，不是 `sft_merge` 本身。
两者的关系是量过的：`sft_merge` 的 Talker 与 `audio_proj` 与父逐位相同，
且它对父的 CER 与 UTMOS 配对判 `indistinguishable`。这条链是全部依据。

### 延迟

| | 实测 | 预算 |
| --- | ---: | ---: |
| 首音（TTF-Audio） | 145.1 ± 4.1 ms，P95 189 | 3000 ms |
| 打断：取消到停 | P50 0.26 ms / P95 0.40 ms（40 轮） | 200 ms |

打断不是软停。取消之后 GPU 占用 0.5 秒内从 41% 掉到 0；同一轮完整跑完要 4953 ms，
取消后墙钟 132 ms。停的是计算本身。

### 产品指标

608 条探针上重算：

| | 本模型 | 父模型 |
| --- | ---: | ---: |
| 中位回复 | 55 字 / 11.8 秒 | 26.0 秒 |
| 占用比 | 6.2× | 13.7× |
| 5 秒时还没说完 | 81% | 93% |
| 空回复 / 循环 | 0 / 1 | |

长度就是这个 checkpoint 存在的理由。另一个候选 `sft_dpo2` 判据全过，
但中位回复念 24.2 秒，而 30 秒级的回复是这个项目会让人不想用第二次的缺陷。

### 对话质量

| | 值 | 状态 |
| --- | ---: | --- |
| 盲评对父（608 条） | 0.6424 ± 0.0468 | 仅报告，注册的位置检查未过（残余偏倚实测 -0.0033） |
| 长度受控对 `sft_len` | 0.5918 ± 0.0664 | 过关，位置 0.5138 与长度偏差 0.5092 都干净 |

判官 `deepseek-ai/DeepSeek-V3.2`，提示词 sha256 与选边种子进产出。
长度被控住、判官实测中性、位置干净之后合并体仍然赢 59.2%，
那只能是质量那个 Δ 的贡献。近乎正交那个预测兑现了。

## 边界

这些是量到底的，判据都写在跑之前。

### 一、"对话建模不退"这一轴，在这个项目的预算内不可认证

对改变长度的干预，我们试了四把尺子，四把都死了，死法各不相同：

| 尺子 | 死因 |
| --- | --- |
| `chat_nll`（原版） | 两个中立作者逐条同号只有 20.3%，机制是参考长度（r=+0.601） |
| 按参考长度分层 | 证伪，控住长度后同号率掉到 1.9% |
| 答得上率 | 连一次已认证的塌方都读不出（+0.0872 落在 ±0.1061 里） |
| `chat_nll`（长度配平版） | 修好了原病灶（同号 94.9%），掉进镜像混淆：分不开"建模更好"和"语域更配" |

要闭合大概需要 3600 条探针，现有 608，差六倍。

选 `sft_merge` 就是选了这个代价：要 12 秒的回复，就得接受这一轴无法认证；
要一条判据全过的交付，就得接受回复仍念 24 秒。

### 二、情绪能力存在，但做不成可控旋钮

参考通路是真的：同一个说话人的两条不同情绪的参考喂进去，输出 F0 +54.7 Hz，
12 个音色全正，零训练。

卡住的是条件化重训的前提。那个前提是标签可信，而四个人的盲听把它证伪了：
人对 `emotion2vec` 标签的一致率 31–42%，人对人 41–65%。两个数都低说明任务本来就难；
人彼此对得上、却都对不上标签，那是标签系统性偏离人耳。条件化 A2A 也真跑过一轮，
主判据 F0 零效应，四条情绪臂对 neutral 全部 `indistinguishable`，
而对照臂先过了，所以是真零不是读不出。

情绪与音色纠缠是学界的公开问题，解法要独立编码器加解耦损失再重训。
架构改动这个项目一开始就排除了。

### 三、打断做到了执行，没做到判断

调用方说停就真停，见上面的 0.26 ms 与 GPU 归零。但谁来判断该停，这一半没有做。
它需要重叠音频加打断标注的语料，我们手上两份语料都没有，合成的重叠只等于 VAD。
端点决定在客户端。

### 其余已量的短板

音色只有 6 个认得出。在成品上重量的 12 选 1 最近邻识别（20 句固定文本）：
serena、eric、uncle_fu、dylan、arthur 是 20/20，moon 17/20；
vivian 16/20、jennifer 6/20 是边缘；cherry、ethan、momo 是 **0/20**，chelsie 1/20，
四个都塌向 serena 和 moon。**后四个不要当独立音色投放。**
完美克隆（参考码回环）是 12/12，所以这是生成侧带的身份不够，不是判官分不开。
推理侧的旋钮救不了：说话人 classifier-free guidance 扫过三档，
四个坏音色没有一个到 15/20，而 moon 反而掉到 10/20。

多轮只测到三轮。不崩、上下文利用 82% 且不衰减，
但答得上的比例随深度掉：42%、24%、14%。

自然度的天花板不在模型。与真人的差距里 79% 是编解码器、21% 是 Talker，
我们已拿到编解码天花板的 92.2%。

## 没有发布的那个候选

`sft_dpo2` 五条判据全过，"对话建模不退"那一轴是认证过的
（同号率 58.9% 够格，两个中立作者 `indistinguishable`），因为它没改长度，
`chat_nll` 对它有资格。它的权重不在这个仓库里。记在这里是因为取舍应当可见：
它换来的是 24.2 秒的中位回复。

## 仓库里有什么

README.md                       ← 这张卡
sft_merge_768.pth        456 MB  成品（Thinker 那半按 fp32 落盘）
tokenizer/               464 KB  tokenizer.json + tokenizer_config.json
configs/                         许可记录、对外数字真源、语音系统提示

中间产物不在这里（`t2a_*`、`sft_dpo*`、`sft_len`）。

## 怎么加载

模型类来自上游 [MiniMind-O](https://github.com/jingyaogong/minimind-o) 的 `model/` 包
（`model_omni.MiniMindOmni`）。这是一个嫁接体，加载有一处不显然的地方：

```python
config = model_omni.OmniConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
```

Thinker 要加宽（`intermediate_size=3584`、`num_key_value_heads=8`），
Talker 必须保持上游默认，两半的配置不同。一起加宽会让 Talker 那 20 个张量形状对不上。
可运行的权威实现在代码仓库的
[`src/mindsurf_omni/service/native.py`](https://github.com/oscar030406/mindsurf-omni/blob/main/src/mindsurf_omni/service/native.py)
里的 `load_native_model`：它会核对参数总数、逐条检查 `missing_keys`，对不上就报错。

音频编解码器另取：[`kyutai/mimi`](https://huggingface.co/kyutai/mimi)，仓库里钉了 revision。

## 许可与署名

不可商用。绑定约束是文本基座，它从训练数据继承 CC-BY-NC-4.0，
这条传导到每一个衍生物，包括本模型。

必须署名 [Mimi](https://huggingface.co/kyutai/mimi)（CC-BY-4.0，钉了 revision）。

## 不在这个仓库里的

语音识别器（SenseVoice）与情绪标注器（emotion2vec）没有随附。
两者都挂 FunASR Model Open Source License Agreement v1.1，
该协议没有明确的再分发条款，所以我们依赖它们、但不替它们分发。
请自行从 ModelScope 获取。

训练语料没有随附，理由同上。

人工盲听材料在另一个仓库：
[oscar0403/mindsurf-omni-listening](https://huggingface.co/datasets/oscar0403/mindsurf-omni-listening)。
