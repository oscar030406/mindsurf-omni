# dpo4/：质量 DPO 第四轮的读数

一句话：**B 过（0.5253 → 0.5684），第三轮那道坎是数据量；但 A 不过、C 作废、D 未跑，不交付。**
全文在 `docs/experiments/2026-08-06-quality-dpo-round4.md`（本地）。

| 文件 | 是什么 |
| --- | --- |
| `params.json` | 配对参数：8 个种子、每提问取 4 对、字数差上限 10、build_seed。**第三轮把这些丢在命令行上了，反推才拿回来** |
| `dpo4-report.json` | 训练：1441 训练 + 160 留出，β 0.1、lr 5e-6、3 epoch，逐 epoch 的留出 `ordered` |
| `B-dpo4-vs-merge.json` | B 的判词、噪声底、座位检查 |
| `blind-dpo4-608.json` | 608 条探针的逐条回复，A 与 E 从这里算 |
| `A-E-dpo4-2026-08-06.json` | A 与 E 的判词，**含 A 的失格证据**（中位数自助 3σ 宽于判据线）与有资格的配对替代读法 |
| `judged-2026-08-06.provenance.json` | 判官身份、端点、提示词哈希 |
| `chat-dpo4-short_{a,b}.json` | C 的原料。**按注册规则未读**（A 不过就不读 C），留着是为了下一轮修好 A 之后不用重新生成 |

排序：先读 `params.json` 知道这一轮怎么建的，再读 `B-dpo4-vs-merge.json` 看结论，
最后读 `A-E-dpo4-2026-08-06.json` 看为什么不交付。
