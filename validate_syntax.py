#!/usr/bin/env python3
"""
Validate Python syntax of benchmark_overhead.py by parsing AST
This can be done WITHOUT executing anything
"""
import ast
import sys

files_to_check = [
    r"s:\Ryot\RYZEN-LLM\scripts\benchmark_overhead.py",
    r"s:\Ryot\RYZEN-LLM\scripts\kernel_optimizer.py",
    r"s:\Ryot\RYZEN-LLM\scripts\semantic_compression.py",
    r"s:\Ryot\RYZEN-LLM\scripts\inference_scaling.py",
]

print("Validating Python syntax by parsing AST...")
print("=" * 60)

for filepath in files_to_check:
    print(f"\nFile: {filepath}")
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse AST
        tree = ast.parse(source)
        print(f"  ✓ Syntax valid")
        
        # Count definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"  Classes: {len(classes)}")
        print(f"  Functions: {len(functions)}")
        
        if classes:
            for cls in classes:
                print(f"    - {cls.name}")
        
    except SyntaxError as e:
        print(f"  ✗ SYNTAX ERROR: {e}")
        print(f"    Line {e.lineno}: {e.text}")
    except FileNotFoundError:
        print(f"  ✗ FILE NOT FOUND")
    except Exception as e:
        print(f"  ✗ ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Validation complete")
