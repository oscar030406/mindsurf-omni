润色训练用的东西。装的是文本和逐条读数，音频不落这里（造对子时在内存里过一遍就丢）。

  pool.jsonl          干净文本池 3180 条：1445 条用户侧问句 + 2116 条长句回复，
                      按正文去重、>160 字的丢掉
  pairs.jsonl         第一轮对子 3169 对（source=转写，target=原文，split=train/val）
  pairs_v2.jsonl      第二轮对子：注入器改成从句内部也放口语词之后重造的
  val_*.jsonl         各臂在留出集上的逐条输出，文件名是「哪个模型 + 哪种解码」

新文件往哪加：造对子进 pairs_*.jsonl，验收的逐条输出进 val_*.jsonl，
汇总数进 artifacts/polish-eval-*.json（不放这里，和别的报告放一起）。
