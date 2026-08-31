"""Tests for the platform layer: tracing and the retrieval-quality gate.

These cover the two properties that matter operationally:

  * telemetry must never be able to break the service. The default path (no
    exporter configured) has to stay a genuine no-op, because that is the path
    CI, local development and any non-Azure host actually run on.

  * the gate must fail when quality drops. A release guard that cannot fail is
    worse than none, because it manufactures confidence -- so the failure case
    is tested directly, not just the passing one.
"""
from __future__ import annotations

import importlib

import pytest

from scripts.gate_retrieval import check


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------

@pytest.fixture
def fresh_telemetry(monkeypatch):
    """A reimported telemetry module with its module-level state reset.

    setup_telemetry() is deliberately idempotent via a module-level flag, so a
    test that wants to exercise configuration has to start from a clean module.
    """
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)
    from app import telemetry
    return importlib.reload(telemetry)


def test_telemetry_is_noop_when_unconfigured(fresh_telemetry):
    """With no exporter configured, setup reports False and spans do nothing."""
    assert fresh_telemetry.setup_telemetry() is False
    assert fresh_telemetry.is_configured() is False

    # The span context manager must still work and yield None.
    with fresh_telemetry.span("test.span", attr=1) as current:
        assert current is None
        # Attaching attributes to a no-op span must not raise.
        fresh_telemetry.set_attributes(current, another=2)


def test_telemetry_configures_with_console_exporter(monkeypatch):
    # OTel lives in requirements-telemetry.txt, not requirements.txt, so CI
    # and the light image run without it. Assert real wiring only where the
    # packages exist; skip elsewhere rather than fail on an intended absence.
    pytest.importorskip("opentelemetry")

    monkeypatch.setenv("OTEL_CONSOLE_EXPORT", "1")
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    from app import telemetry
    mod = importlib.reload(telemetry)

    assert mod.setup_telemetry() is True
    assert mod.is_configured() is True
    with mod.span("test.span", k=3) as current:
        assert current is not None

    # Leave global tracing state as we found it for the rest of the suite.
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)
    importlib.reload(telemetry)


def test_telemetry_survives_a_broken_exporter(monkeypatch):
    """A malformed connection string must not take the service down."""
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "this-is-not-valid")
    monkeypatch.delenv("OTEL_CONSOLE_EXPORT", raising=False)
    from app import telemetry
    mod = importlib.reload(telemetry)

    # Either it fails and degrades to a no-op, or it accepts the string and
    # fails later on export. Both are fine; raising here is not.
    result = mod.setup_telemetry()
    assert result in (True, False)
    with mod.span("test.span"):
        pass

    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    importlib.reload(telemetry)


def test_answering_still_works_with_telemetry_off(index):
    """The traced code path must behave identically when tracing is disabled."""
    from app import rag
    result = rag.answer_question(index, "spindle running too hot", k=3)
    assert result.grounded is True
    assert result.sources


# --------------------------------------------------------------------------
# Retrieval gate
# --------------------------------------------------------------------------

BASELINE = {
    "n_questions": 30,
    "mrr": 0.8528,
    "hit@1": 0.7667,
    "recall@1": 0.6333,
    "hit@3": 0.9333,
    "recall@3": 0.8333,
    "hit@5": 0.9667,
    "recall@5": 0.8833,
}

FLOORS = {"min_mrr": 0.75, "min_hit3": 0.85, "max_drop": 0.05}


def test_gate_passes_on_the_baseline_itself():
    assert check(dict(BASELINE), BASELINE, **FLOORS) == []


def test_gate_fails_below_the_absolute_floor():
    current = dict(BASELINE, mrr=0.70)
    failures = check(current, None, **FLOORS)
    assert any("floor" in f for f in failures)


def test_gate_fails_on_regression_against_baseline():
    """Still above the absolute floor, but clearly worse than the baseline."""
    current = dict(BASELINE, mrr=0.78)   # 0.75 floor is met, but -0.07 vs baseline
    failures = check(current, BASELINE, **FLOORS)
    assert any("dropped" in f for f in failures)


def test_gate_tolerates_noise_within_tolerance():
    current = dict(BASELINE, mrr=0.8428)  # -0.01, inside the 0.05 tolerance
    assert check(current, BASELINE, **FLOORS) == []


def test_gate_catches_regression_in_a_non_headline_metric():
    """MRR flat, recall@5 collapsed -- the gate must still fail."""
    current = dict(BASELINE, **{"recall@5": 0.60})
    failures = check(current, BASELINE, **FLOORS)
    assert any("recall@5" in f for f in failures)


def test_gate_allows_improvement():
    current = dict(BASELINE, mrr=0.95, **{"hit@3": 0.99})
    assert check(current, BASELINE, **FLOORS) == []
