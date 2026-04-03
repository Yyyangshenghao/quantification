from __future__ import annotations

from src.strategy.metric_map import bucket_for_industry, bucket_for_industry_optional, metric_for_industry, metric_for_industry_optional


def test_bucket_and_metric_map_from_config(configs: dict) -> None:
    metric_map_cfg = configs["metric_map"]
    assert bucket_for_industry("银行", metric_map_cfg) == "defensive_dividend"
    assert metric_for_industry("银行", metric_map_cfg) == "pb"
    assert bucket_for_industry("煤炭", metric_map_cfg) == "cyclical_rotation"
    assert metric_for_industry("煤炭", metric_map_cfg) == "pb"
    assert bucket_for_industry_optional("未知行业", metric_map_cfg) is None
    assert metric_for_industry_optional("未知行业", metric_map_cfg) is None
