# artifacts/：每个对外数字的逐条读数

仓库根 `README.md` 和两张卡里的每一个数字，源头都在这里。
`configs/release/headline_numbers.json` 的 `source` 字段逐条指过来。

**这里只放读数，不放叙事。** 一份 json 是一次测量的产出：判词、噪声底、
样本量、判官身份、种子。想知道某个数是怎么算出来的，读产生它的脚本，
不是读这里的文件名。

## 什么在库里、什么不在

| | 在库里 | 不在 |
| --- | --- | --- |
| 逐条读数 | 295 个文件、约 13 MB：`.json` 判词与报告、`.jsonl` 逐样本、`.xlsx`/`.csv` 评分表 | |
| 音频 | | `.wav` 全部挡在 `.gitignore` 外。2691 个、2.3 GB，留在生成它的机器上 |
| 权重 | | `.pth` 同理 |

**没有音频也能重算判定**：CER、UTMOS 这些是逐样本算好写进 `.jsonl` 的，
比较脚本读的是那些数不是音频。重新转写或重打 UTMOS 才需要音频，那要重新生成，要卡。

一个例外入库了：`reference-logits-*.npz`。它是「基座移植是否忠实」的唯一判据，
`tests/test_conversion_tolerance.py` 要用。

## 怎么排

根目录 146 个文件是**单次测量**，命名是 `<测什么>-<对象>-<日期>.json`。
子目录是**成组的一批**，一批里的文件互相要一起读：

| 目录 | 装什么 |
| --- | --- |
| `merge/` | 成品 `sft_merge` 的语音、克隆、对话读数 |
| `blind608/` | 608 条探针的盲评：逐对判词、聚合胜率、各臂回复 |
| `short_refs/` | 长度配平参考集上的打分（那把尺子没过考试，文件留着没出判词） |
| `listening_models/` `listening_synthesiser/` `listening_emotion/` | 三个盲听包的揭盲表 `key.json` |
| `listening_returned/` | 评分员交回的十二份表 |
| `barge_in/` | 打断的 40 轮读数，加「计算真的停了」那份墙钟与 GPU 证明 |
| `emotion-gate/` `emotion-conditioned/` `emotion-harvest/` `emotion-pack/` | 情绪四条路各自的读数 |
| `chat-matrix/` | 每个臂给每个臂的文本打分，对角线自我偏好就在这张表里 |
| `judge-length-sensitivity/` | 判官对回答长度有多敏感：900 对按字数差分三箱的逐对判词、聚合报告、判官出处 |
| `judge-reliability/` | 同一批 900 对左右对调后重判：翻转率、翻转的形状（噪声还是左座偏好）。要和上一行一起读，`of_key` 指回去 |
| `dpo/` | DPO 各轮的训练与验收读数 |
| 其余按臂命名的目录 | 该臂生成的一批样本的转写与打分 |

## 新文件往哪加

一次测量一个文件，日期进文件名。成组的开子目录。
**产出要能自证**：判官是谁、种子是多少、噪声底多少，写进 json 而不是写在旁边的文档里，
因为文档不进这个仓库而这个文件会。

从服务器拉证据回来先跑一遍 `python scripts/scrub_artifact_paths.py --check <文件>`：
归档里带过生成机器的个人目录，3059 处。
