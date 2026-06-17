"""Tests for LBO debt amortization schedule."""

import pytest

from valuationengine.core import debt


def test_schedule_columns():
    """Debt schedule has all required columns."""
    sched = debt.build_schedule(
        initial_debt=100.0,
        interest_rate=0.08,
        mandatory_amortization_pct=0.05,
        available_cash_flow=[20.0, 22.0, 24.0, 26.0, 28.0],
        cash_sweep_pct=0.75,
        years=5,
    )
    for col in [
        "year",
        "beginning_balance",
        "interest_expense",
        "mandatory_paydown",
        "sweep_paydown",
        "ending_balance",
    ]:
        assert col in sched.columns


def test_schedule_length():
    """Schedule has one row per year."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [10.0] * 7, 0.75, 7)
    assert len(sched) == 7


def test_initial_beginning_balance():
    """Year 1 beginning_balance equals initial_debt."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [10.0] * 5, 0.75, 5)
    assert sched.iloc[0]["beginning_balance"] == pytest.approx(100.0)


def test_year_to_year_continuity():
    """Year N+1 beginning_balance equals year N ending_balance."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [20.0] * 5, 0.75, 5)
    for i in range(1, len(sched)):
        assert sched.iloc[i]["beginning_balance"] == pytest.approx(
            sched.iloc[i - 1]["ending_balance"]
        )


def test_interest_expense_formula():
    """Interest expense equals beginning_balance times interest_rate."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [20.0] * 5, 0.75, 5)
    assert sched.iloc[0]["interest_expense"] == pytest.approx(100.0 * 0.08)


def test_balance_never_negative():
    """Ending balance can hit zero but never go below."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [1000.0] * 5, 1.0, 5)
    assert (sched["ending_balance"] >= 0).all()


def test_zero_cash_flow_only_mandatory_paydown():
    """If cash flow is zero, only mandatory paydown reduces the balance."""
    sched = debt.build_schedule(100.0, 0.08, 0.05, [0.0] * 5, 0.75, 5)
    assert sched.iloc[0]["sweep_paydown"] == pytest.approx(0.0)
    assert sched.iloc[0]["mandatory_paydown"] == pytest.approx(5.0)
