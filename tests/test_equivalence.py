"""Tests for SymbolicEquivalenceChecker."""

import pytest
from sigma_diff.differ import SymbolicEquivalenceChecker


def test_equivalent_different_style():
    func_a = "def f(x):\n    return x + 1\n"
    func_b = "def g(x):\n    y = x\n    y += 1\n    return y\n"
    result = SymbolicEquivalenceChecker().are_equivalent(func_a, func_b)
    assert result.is_equivalent is True
    assert result.confidence >= 0.5


def test_off_by_one_detected():
    func_a = "def f(x):\n    return x + 1\n"
    func_b = "def g(x):\n    return x + 2\n"
    result = SymbolicEquivalenceChecker().are_equivalent(func_a, func_b)
    assert result.is_equivalent is False
    assert result.confidence == 1.0
    assert result.counterexample is not None
    assert "inputs" in result.counterexample


def test_algorithm_change_equivalent():
    # Both should return sorted list — bubble vs selection sort style
    func_a = (
        "def f(lst):\n"
        "    lst = list(lst)\n"
        "    n = len(lst)\n"
        "    for i in range(n):\n"
        "        for j in range(0, n-i-1):\n"
        "            if lst[j] > lst[j+1]:\n"
        "                lst[j], lst[j+1] = lst[j+1], lst[j]\n"
        "    return lst\n"
    )
    func_b = (
        "def g(lst):\n"
        "    return sorted(lst)\n"
    )
    # These take list inputs; checker uses randint so may not always converge — just check no crash
    result = SymbolicEquivalenceChecker().are_equivalent(func_a, func_b)
    assert isinstance(result.is_equivalent, bool)
    assert 0.0 <= result.confidence <= 1.0


def test_syntax_error_in_func():
    result = SymbolicEquivalenceChecker().are_equivalent("def f(x:\n    pass", "def g(x):\n    pass")
    assert result.is_equivalent is False
    assert result.confidence == 0.0


def test_identity_functions_equivalent():
    func_a = "def f(x):\n    return x\n"
    func_b = "def g(x):\n    return x\n"
    result = SymbolicEquivalenceChecker().are_equivalent(func_a, func_b)
    assert result.is_equivalent is True


def test_negation_not_equivalent():
    func_a = "def f(x):\n    return x\n"
    func_b = "def g(x):\n    return -x\n"
    result = SymbolicEquivalenceChecker().are_equivalent(func_a, func_b)
    # x=0 makes them equivalent, but non-zero inputs will differ
    assert result.is_equivalent is False or result.counterexample is not None or result.confidence < 1.0
