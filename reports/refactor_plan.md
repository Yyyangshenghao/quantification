# Refactor Plan

1. 配置与 universe：
   修改 `config/strategy.yml`、`config/metric_map.yml`、`config/universe.yml`、`README.md`、`AGENTS.md`，新增 `config/universe_rules.yml`。
2. universe 生成器：
   修改 `scripts/refresh_universe.py`、`src/strategy/universe.py`、`src/strategy/quality.py`、`src/pipeline/features.py`。
3. snapshot 与动作引擎：
   修改 `scripts/prepare_snapshot.py`、`src/pipeline/snapshot.py`、`src/strategy/fundamentals.py`、`src/strategy/signals.py`、`src/strategy/regime.py`。
4. 报告与执行清单：
   修改 `scripts/render_report.py`、`src/reporting/render.py`、`prompts/daily_decision.md`。
5. 回测适配：
   修改 `src/strategy/backtest_engine.py`，让 decision scope / frozen holdings / action enum 与日常流程一致。
6. 测试：
   更新 `tests/test_signals.py`、`tests/test_backtest_execution.py`、`tests/test_quality_and_cycle.py`、`tests/test_fundamental_break.py`。
7. 新增测试：
   增加 universe 重建、hysteresis、decision scope、safe mode、orders 输出、prompt 只解释不裁决等测试文件。

## 行为变更

- `config/universe.yml` 从手工白名单改为自动生成的 effective universe 产物。
- `refresh_universe.py` 从建议候选池改为可按月/季生成并可 `--apply` 生效的正式 universe 生成器。
- 每日决策范围从仅看有效池改为 `effective_universe ∪ current_holdings`。
- 持仓引入 `ACTIVE / FROZEN / FORCE_EXIT` 状态机；被踢出池子先冻结，不直接清仓。
- 最终动作改为固定枚举，由 Python 规则引擎决定；prompt 只做解释。
- 新增确定性仓位分配器、safe mode、orders JSON/CSV、universe diff 与审计字段。

## 测试变更

- 覆盖 monthly rebuild、rank hysteresis、dropped holding -> FROZEN、FROZEN 不可加仓、FORCE_EXIT 必卖。
- 覆盖 decision scope 并集、priority 分配稳定性、safe mode 阻止新买入。
- 覆盖 orders 输出字段完整性、prompt 不改 `action_enum`、回测兼容 frozen/effective universe。
