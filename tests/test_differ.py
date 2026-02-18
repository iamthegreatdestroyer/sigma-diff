"""Tests for sigma-diff semantic diffing engine."""

from sigma_diff.differ import (
    ChangeType,
    SemanticDiffer,
    StructuralElement,
    _classify_line,
)
from sigma_diff.scorer import compute_semantic_score, ImpactLevel, element_impact
from sigma_diff.summary import summarize_diff


def test_classify_python_function():
    assert _classify_line("def foo():", ".py") == StructuralElement.FUNCTION


def test_classify_python_class():
    assert _classify_line("class MyClass:", ".py") == StructuralElement.CLASS


def test_classify_python_method():
    assert _classify_line("    def bar(self):", ".py") == StructuralElement.METHOD


def test_classify_python_import():
    assert _classify_line("import os", ".py") == StructuralElement.IMPORT
    assert _classify_line("from os import path", ".py") == StructuralElement.IMPORT


def test_classify_rust_function():
    assert _classify_line("pub fn main() {", ".rs") == StructuralElement.FUNCTION
    assert _classify_line("fn helper() {", ".rs") == StructuralElement.FUNCTION


def test_classify_go_function():
    assert _classify_line("func main() {", ".go") == StructuralElement.FUNCTION


def test_classify_whitespace():
    assert _classify_line("", ".py") == StructuralElement.WHITESPACE
    assert _classify_line("   ", ".py") == StructuralElement.WHITESPACE


def test_diff_texts_added():
    differ = SemanticDiffer()
    fc = differ.diff_texts("", "def hello():\n    pass\n", path="new.py")
    assert fc.change_type == ChangeType.ADDED
    assert fc.additions > 0


def test_diff_texts_deleted():
    differ = SemanticDiffer()
    fc = differ.diff_texts("def goodbye():\n    pass\n", "", path="old.py")
    assert fc.change_type == ChangeType.DELETED
    assert fc.deletions > 0


def test_diff_texts_modified():
    differ = SemanticDiffer()
    old = "def foo():\n    return 1\n"
    new = "def foo():\n    return 2\n"
    fc = differ.diff_texts(old, new, path="mod.py")
    assert fc.change_type == ChangeType.MODIFIED
    assert fc.additions >= 1
    assert fc.deletions >= 1


def test_diff_texts_structural_elements():
    differ = SemanticDiffer()
    old = "x = 1\n"
    new = "x = 1\ndef new_func():\n    pass\n"
    fc = differ.diff_texts(old, new, path="add_func.py")
    assert StructuralElement.FUNCTION in fc.structural_elements
    assert fc.is_api_change


def test_diff_texts_comment_only():
    differ = SemanticDiffer()
    old = "x = 1\n"
    new = "# added a comment\nx = 1\n"
    fc = differ.diff_texts(old, new, path="comment.py")
    assert StructuralElement.COMMENT in fc.structural_elements


def test_rename_detection():
    differ = SemanticDiffer()
    content = "def foo():\n    return 42\n"
    fc = differ.diff_texts(content, content, path="new_name.py", old_path="old_name.py")
    assert fc.change_type == ChangeType.RENAMED


def test_semantic_score_high_impact():
    differ = SemanticDiffer()
    old = "def critical_api():\n    return 1\n"
    new = "def critical_api(new_param):\n    return new_param + 1\n"
    fc = differ.diff_texts(old, new, path="api.py")
    score = compute_semantic_score([fc])
    assert score > 0.5


def test_semantic_score_low_impact():
    differ = SemanticDiffer()
    old = "# old comment\nx = 1\n"
    new = "# new comment\nx = 1\n"
    fc = differ.diff_texts(old, new, path="comments.py")
    score = compute_semantic_score([fc])
    assert score < 0.5


def test_semantic_score_empty():
    assert compute_semantic_score([]) == 0.0


def test_element_impact_levels():
    assert element_impact("function") == ImpactLevel.CRITICAL
    assert element_impact("class") == ImpactLevel.CRITICAL
    assert element_impact("comment") == ImpactLevel.LOW
    assert element_impact("whitespace") == ImpactLevel.LOW


def test_summarize_diff():
    differ = SemanticDiffer()
    old = "def foo():\n    return 1\n"
    new = "def foo():\n    return 2\ndef bar():\n    pass\n"
    fc = differ.diff_texts(old, new, path="test.py")
    from sigma_diff.differ import DiffResult
    result = DiffResult(
        files=[fc],
        total_additions=fc.additions,
        total_deletions=fc.deletions,
        semantic_score=compute_semantic_score([fc]),
    )
    summary = summarize_diff(result)
    assert "1 file(s) changed" in summary
    assert "test.py" in summary
