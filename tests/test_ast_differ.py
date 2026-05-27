"""Tests for ASTDiffer."""

import pytest
from sigma_diff.differ import ASTDiffer, NodeChange


def test_identical_functions():
    code = "def foo(x):\n    return x + 1\n"
    result = ASTDiffer().diff(code, code)
    assert result.structural_similarity == 1.0
    assert result.additions == []
    assert result.deletions == []
    assert result.node_changes == []


def test_renamed_variable():
    code_a = "def foo(x):\n    result = x + 1\n    return result\n"
    code_b = "def foo(x):\n    value = x + 1\n    return value\n"
    result = ASTDiffer().diff(code_a, code_b)
    assert result.structural_similarity > 0.8


def test_added_param():
    code_a = "def foo(x):\n    return x\n"
    code_b = "def foo(x, y):\n    return x + y\n"
    result = ASTDiffer().diff(code_a, code_b)
    changes = [nc for nc in result.node_changes if nc.node_name == "foo"]
    assert any(nc.change_type == "modified" for nc in changes)


def test_added_function():
    code_a = "def foo(x):\n    return x\n"
    code_b = "def foo(x):\n    return x\n\ndef bar(y):\n    return y\n"
    result = ASTDiffer().diff(code_a, code_b)
    assert "bar" in result.additions


def test_deleted_function():
    code_a = "def foo(x):\n    return x\n\ndef bar(y):\n    return y\n"
    code_b = "def foo(x):\n    return x\n"
    result = ASTDiffer().diff(code_a, code_b)
    assert "bar" in result.deletions


def test_syntax_error_graceful():
    result = ASTDiffer().diff("def f(:", "def g(): pass")
    assert result.structural_similarity == 0.0
    assert result.additions == []
    assert result.deletions == []


def test_both_syntax_errors_graceful():
    result = ASTDiffer().diff("def f(:", "class {bad:")
    assert result.structural_similarity == 0.0


def test_node_change_added_recorded():
    code_a = ""
    code_b = "def new_func():\n    pass\n"
    result = ASTDiffer().diff(code_a, code_b)
    assert "new_func" in result.additions
    nc = next((c for c in result.node_changes if c.node_name == "new_func"), None)
    assert nc is not None
    assert nc.change_type == "added"
    assert nc.old_value is None
