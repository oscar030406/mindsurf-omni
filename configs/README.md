# configs/：探针集、参照集、以及对外的四份定稿

## release/ 是给外面看的

| 文件 | 是什么 |
| --- | --- |
| `headline_numbers.json` | **对外数字的唯一真源**。README、模型卡、答辩材料一律从它取数，每条带 `source` 指向 `artifacts/` 里的读数 |
| `licence.json` | 逐条许可记录，含四项未核实的和「要怎样才能改掉这个结论」 |
| `MODEL_CARD.md` | Hugging Face 模型仓的 README |
| `LISTENING_DATASET_CARD.md` | Hugging Face 盲听数据集的 README |

改数字要连 `source` 一起改。同一个数散落六处、彼此差 0.001，是评审最容易抓住的破绽。

## 探针集：问模型的那些话

| 文件 | 条数 | 用来测什么 |
| --- | ---: | --- |
| `speech_probes_zh_v1.jsonl` | 160 | 语音：固定文本 teacher-forcing，CER / UTMOS / 静音率 |
| `talker_texts_zh_v1.jsonl` | 160 | Talker 隔离，与上面同一批文本 |
| `blind_probes_zh_v1.jsonl` | 608 | 对话盲评。158 条不够用，扩到 608 才够分辨主判据 |
| `chat_probes_zh_v2.jsonl` | 450 | 对话似然 |
| `preference_prompts_zh_all.jsonl` | 838 | DPO 训练提示 |
| `multiturn_probes_zh_v1.jsonl` | 50 | 多轮，第 2、3 轮是省略句，不看历史答不了 |
| `emotion_probes_zh_*.jsonl` | 五类 | 情绪条件化 |
| `realtime_probes_zh_v1.jsonl` | | 延迟与打断 |
| `talker_clauses_zh_v1.jsonl` | | 合成器吐块 |

## 每个参照集必须带 `.provenance.json`

它记的是**谁写的这批参考回复**。这件事事后推断不出来——实测过，
采样温度下真作者对自己旧回复的相似度均值只有 0.291，所以它是**声明**不是推断。

工具会拦：作者是被比较的臂之一就拒绝比较。这条规矩是花代价买的——
曾经有一批「holdout」是被比较的那个模型自己生成的，四条对角线四个最小值，
自我偏好最大 0.83 nat，是阈值的 16 倍。

`chat_refs_short_a/b_v1` 是两个新作者写的短语域参考，
它们对**不改长度**的干预是目前最强的对话仪器（两作者逐条同号 94.9–100%）。

## `system_prompt_speech_zh.txt`

语音路径的系统提示。**试过用它压短回复，是负增益**：
中位 30.0 到 30.9 秒纹丝不动，而编造从 13% 涨到 42%。这个规模的模型指令跟随太弱，
长度只能靠训练改。文件留着是因为它是当前生效的那份，不是因为它有效。
