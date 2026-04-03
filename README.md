# A 股半自动规则研究项目

这是一个面向 A 股 long-only 的半自动研究工程。系统只生成规则驱动的股票池、每日动作建议和手动执行清单；不接券商、不自动交易、不把买卖裁决交给 LLM。

## 当前原则

- Python 规则引擎决定 `action_enum`、目标仓位、目标金额。
- LLM 只允许解释已有结果、检查异常、辅助审阅，不允许重算或改写 `action_enum`。
- `config/universe.yml` 是自动生成的 effective universe 产物，不再手工编辑。
- `config/positions.yml` 继续手工维护，表示真实账户持仓。
- `config/account.yml` 用于金额与仓位货币化约束；缺失时只输出方向性建议。

## 关键对象

### candidate pool

通过硬过滤并完成打分的候选集合。

### effective universe

当前再平衡周期允许新开仓/加仓的正式有效池，由 `scripts/refresh_universe.py --apply` 自动生成。

### current holdings scope

真实持仓范围。每日决策作用域始终是：

`effective_universe ∪ current_holdings`

因此，持仓即使已被踢出有效池，也不会从日常风控里消失。

## universe 生成逻辑

- 默认频率：月度；也支持季度。
- 再平衡日：月末或季末收盘后生成新一期 effective universe，下一交易日生效。
- 非再平衡日：复用当前 effective universe，不重建。
- 稳定器：老成员只要仍过硬过滤且行业内排名 `<= 4` 可保留；新成员只有行业内排名 `<= 2` 才能进入。
- 行业上限：每个行业最多 2 只。
- 输出：
  - `config/universe.yml`
  - `reports/universe/latest.md`
  - `reports/universe/latest.json`
  - `data/curated/universe_history/`

## 持仓状态机

### ACTIVE

当前在 effective universe 中。允许 `BUY_1 / BUY_2 / BUY_3 / HOLD / REDUCE / SELL_ALL`。

### FROZEN

当前不在 effective universe 中，但真实持仓仍在。只允许 `HOLD_FROZEN / REDUCE / SELL_ALL`。

被踢出池子不会立刻卖出，因为“失去入池资格”不等于“必须立即退出”。系统先冻结，继续做风险控制与减仓判断，避免月月大换血。

### FORCE_EXIT

命中 ST、fundamental break、退市风险、长期数据失真、长期停牌等硬规则。必须 `SELL_ALL`。

## 账户与金额约束

- 回测读取 `config/account.yml -> account.initial_capital`，因为回测需要从明确初始资金开始模拟净值与仓位演化。
- 日常执行读取 `current_cash`、`reserved_cash`、`latest_total_equity`，因为现实账户的可买金额取决于当前现金和当前总权益，而不是历史初始资金。
- `target_position_tranches -> target_weight` 的映射由 `config/account.yml -> position_sizing.tranche_weights` 决定。
- 若缺少 `account.yml` 或关键字段缺失，系统仍输出 `action_enum`，但 orders 会进入 degraded mode：
  - 不输出精确 `target_order_value`
  - 报告显式提示“仅有方向性建议，未完成金额约束”

## 日度动作

固定动作枚举：

- `BUY_1`
- `BUY_2`
- `BUY_3`
- `HOLD`
- `HOLD_FROZEN`
- `REDUCE`
- `SELL_ALL`
- `EMPTY`
- `BLOCKED`
- `DATA_ERROR`

每日 orders 至少包含：

- 当前/目标 tranche
- 当前/目标权重
- 目标仓位变化
- `target_order_value`
- `priority_score`
- `blocked_reason`
- `reason_codes`
- `risk_flags`

## 运行流程

### 1. 更新数据

```bash
./.venv/bin/python scripts/update_market_data.py --start-date 2016-01-01 --end-date 2026-04-03 --all-stocks
```

### 2. 构建特征

```bash
./.venv/bin/python scripts/build_features.py
```

### 3. 重建 effective universe

```bash
./.venv/bin/python scripts/refresh_universe.py --as-of-date 2026-04-03 --apply
```

### 4. 生成 snapshot

```bash
./.venv/bin/python scripts/prepare_snapshot.py --as-of-date 2026-04-03
```

### 5. 生成 orders 与日报

```bash
./.venv/bin/python scripts/render_report.py --snapshot-json data/snapshots/latest.json
```

固定输出：

- `reports/daily/latest.json`
- `reports/daily/latest.md`
- `reports/daily/orders_latest.json`
- `reports/daily/orders_latest.csv`

## 回测说明

- 回测复用同一套 action enum、仓位段数和 tranche weight。
- 若特征中存在历史 effective universe / holding_state 信息，回测会沿用该范围与冻结逻辑。
- 若历史 effective universe 不完整，回测至少仍会保留已持仓标的的风险控制逻辑，不把“脱池”直接等同于“立刻清仓”。

## 数据源

- 免费数据源优先。
- `JQData` 仅在环境变量存在时启用。
- 数据源失败时必须显式降级，不允许静默填假数据。

## 约束

- 禁止自动交易、自动下单、券商接入。
- 规则长期写在代码与配置中，不允许在 prompt 中临场决定交易动作。
- 修改策略阈值时必须同步更新配置、README 与测试。
