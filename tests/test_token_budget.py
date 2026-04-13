"""Tests for the token_budget measurement helper."""

import logging

from src.agent.token_budget import count_tokens, measure


def test_count_tokens_empty_returns_zero() -> None:
    assert count_tokens("") == 0


def test_count_tokens_scales_with_safety_factor() -> None:
    text = "Hello world, this is a small test sentence."
    raw = count_tokens(text, safety_factor=1.0)
    scaled = count_tokens(text, safety_factor=2.0)
    assert raw > 0
    # Scaled should be roughly double — allow 1 unit slack from rounding.
    assert abs(scaled - raw * 2) <= 1


def test_count_tokens_handles_multilingual() -> None:
    danish = "Hvad er reglerne for eksamen på Københavns Universitet?"
    chinese = "学生考试规则是什么？"
    assert count_tokens(danish) > 0
    assert count_tokens(chinese) > 0


def test_measure_disabled_returns_zero_and_no_log(caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.INFO, logger="src.agent.token_budget"):
        result = measure("planner", "some prompt text", enabled=False)
    assert result == 0
    assert not any("token_budget" in rec.message for rec in caplog.records)


def test_measure_enabled_logs_and_returns_count(caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.INFO, logger="src.agent.token_budget"):
        result = measure("planner", "Hello world", enabled=True)
    assert result > 0
    assert any(
        "token_budget" in rec.message and "planner" in rec.message
        for rec in caplog.records
    )
