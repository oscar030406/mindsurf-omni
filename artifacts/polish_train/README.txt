润色训练用的东西。装的是文本和逐条读数，音频不落这里（造对子时在内存里过一遍就丢）。

  pool.jsonl              干净文本池 3180 条：1445 条用户侧问句 + 2116 条长句回复，
                          按正文去重、>160 字的丢掉
  pool_holdout.jsonl      另外 988 条，训练池一条没取过（按正文去重）。任何 checkpoint
                          都没见过它们，所以磁盘上的臂可以直接在上面重读，不用重训
  pairs.jsonl             第一轮对子 3169 对（source=转写，target=原文，split=train/val）
  pairs_v2.jsonl          第二轮：注入器改成从句内部也放口语词之后重造的
  pairs_v3.jsonl          第三轮：目标换成「转写去掉注入段」，原目标留在 target_corpus
  pairs_holdout.jsonl     pool_holdout 造的对子，全部标 val
  val_*.jsonl             各臂在留出集上的逐条输出，文件名是「哪个模型 + 哪种解码」

只入库模型输出。离线算得出来的不入库——脚本是确定性的，秒级重跑：

  val_bigholdout_ceiling / _nothing      scripts/polish_ceiling.py --mode perfect / nothing
  val_bigholdout_{union,intersection,
    vocabunion,veto}_t*.jsonl            scripts/merge_polish_arms.py --mode ...
  dict_*.jsonl / nondict_*.jsonl         scripts/filter_dictation_register.py
  pairs_v4.jsonl                         scripts/filter_dictation_register.py --by-pool-text
  pairs_v3_0.*.jsonl                     scripts/subsample_polish_pairs.py --fraction ...

汇总数一律入库（artifacts/polish-eval-*.json、polish-frontier-*.json），
因为那才是被引用的数字，而且小。

新文件往哪加：造对子进 pairs_*.jsonl，模型验收的逐条输出进 val_*.jsonl，
汇总进 artifacts/polish-eval-*.json（不放这里，和别的报告放一起）。
