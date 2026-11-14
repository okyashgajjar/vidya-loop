"""Unit tests for LangGraph conditional edge routing decisions."""

import pytest
from src.graph import routing_decision
from src.schemas import AgentState


def test_routing_native_failure_attempt_1_retries():
    state: AgentState = {
        "native_eval_passed": False,
        "attempt": 1,
    }
    assert routing_decision(state) == "retry_generate"


def test_routing_native_failure_attempt_3_terminates():
    state: AgentState = {
        "native_eval_passed": False,
        "attempt": 3,
    }
    assert routing_decision(state) == "ship_failure"


def test_routing_native_pass_proceeds_to_judge():
    state: AgentState = {
        "native_eval_passed": True,
        "judge_eval_passed": None,
        "attempt": 1,
    }
    assert routing_decision(state) == "eval_judge"


def test_routing_both_pass_ships_success():
    state: AgentState = {
        "native_eval_passed": True,
        "judge_eval_passed": True,
        "attempt": 1,
    }
    assert routing_decision(state) == "ship_success"


def test_routing_judge_failure_attempt_2_retries():
    state: AgentState = {
        "native_eval_passed": True,
        "judge_eval_passed": False,
        "attempt": 2,
    }
    assert routing_decision(state) == "retry_generate"


def test_routing_stops_when_judge_cannot_evaluate():
    state: AgentState = {
        "native_eval_passed": True,
        "judge_eval_passed": False,
        "judge_unavailable": True,
        "attempt": 1,
    }
    assert routing_decision(state) == "ship_failure"


def test_routing_judge_failure_attempt_3_terminates():
    state: AgentState = {
        "native_eval_passed": True,
        "judge_eval_passed": False,
        "attempt": 3,
    }
    assert routing_decision(state) == "ship_failure"

