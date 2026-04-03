from __future__ import annotations

from src.utils.exceptions import ConfigError


def candidate_buckets_for_industry(industry: str, metric_map_cfg: dict) -> list[str]:
    candidates = metric_map_cfg.get("industry_bucket_candidates", {}).get(industry)
    if candidates:
        return list(candidates)
    return [bucket_for_industry(industry, metric_map_cfg)]


def bucket_for_industry(industry: str, metric_map_cfg: dict) -> str:
    bucket = metric_map_cfg.get("industry_bucket_map", {}).get(industry)
    if not bucket:
        raise ConfigError(f"Industry bucket mapping missing for {industry}")
    return bucket


def metric_for_industry(industry: str, metric_map_cfg: dict) -> str:
    metric = metric_map_cfg.get("industry_metric_map", {}).get(industry)
    if metric:
        return metric
    bucket = bucket_for_industry(industry, metric_map_cfg)
    fallback = metric_map_cfg.get("bucket_default_metric", {}).get(bucket)
    if not fallback:
        raise ConfigError(f"Metric mapping missing for industry={industry}, bucket={bucket}")
    return fallback


def is_financial_industry(industry: str, metric_map_cfg: dict) -> bool:
    return industry in set(metric_map_cfg.get("financial_industries", []))
