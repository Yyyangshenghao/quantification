# Repository Rules

- 本仓库禁止接入任何自动交易、自动下单、券商 API 或委托执行能力。
- `config/universe.yml` 是自动生成的 effective universe 产物，只能由程序在 `refresh_universe.py --apply` 时更新。
- `data/ledger/trades.csv` 是成交账本；`config/positions.yml` 建议由 `scripts/reconcile_positions.py` 重建，不建议手工逐笔编辑。
- 长期规则必须写在代码和配置里；LLM 只允许解释、检查和审阅结果，不允许改动 `action_enum`。
- `prompts/daily_decision.md` 只能解释 snapshot / orders 输出，不得在 prompt 中新增策略裁决逻辑。
- 修改任何策略阈值时，必须同步更新 `config/strategy.yml`、`config/metric_map.yml`、`README.md` 与相关测试。
- 修改回测逻辑前后，必须先运行并通过测试，再更新回测相关文档。
- 遇到数据源失败时优先降级到备用适配器，并在输出中显式记录降级情况；禁止默默填充假数据。
- 所有研究输出、日报、回测结果都必须可以追溯到本地 snapshot、本地缓存文件、自动生成的 orders 文件与 `data/curated/run_manifest.json`。
