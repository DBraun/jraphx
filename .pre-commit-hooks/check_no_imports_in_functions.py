#!/usr/bin/env python3
"""
Pre-commit hook to detect import statements within functions.
"""
import argparse
import ast
import sys
from pathlib import Path


class ImportInFunctionChecker(ast.NodeVisitor):
    """AST visitor to find import statements within function definitions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.errors: list[tuple[int, str]] = []
        self.in_function = False
        self.function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition nodes."""
        self.in_function = True
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
        self.in_function = len(self.function_stack) > 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition nodes."""
        self.in_function = True
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()
        self.in_function = len(self.function_stack) > 0

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import nodes."""
        if self.in_function:
            function_name = ".".join(self.function_stack)
            modules = ", ".join(alias.name for alias in node.names)
            self.errors.append(
                (
                    node.lineno,
                    f"Import statement found in function '{function_name}': import {modules}",
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import nodes."""
        if self.in_function:
            function_name = ".".join(self.function_stack)
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            self.errors.append(
                (
                    node.lineno,
                    f"Import statement found in function '{function_name}': from {module} import {names}",
                )
            )
        self.generic_visit(node)


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Check a single Python file for imports within functions."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ImportInFunctionChecker(str(filepath))
        checker.visit(tree)
        return checker.errors

    except SyntaxError:
        # Skip files with syntax errors (they'll be caught by other tools)
        return []
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return []


def main() -> int:
    """Main entry point for the pre-commit hook."""
    parser = argparse.ArgumentParser(description="Check for import statements within functions")
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args()

    exit_code = 0

    for filename in args.filenames:
        filepath = Path(filename)

        # Only check Python files
        if filepath.suffix != ".py":
            continue

        errors = check_file(filepath)

        if errors:
            exit_code = 1
            print(f"\n{filename}:")
            for line_no, message in errors:
                print(f"  Line {line_no}: {message}")

    if exit_code == 1:
        print("\nImports within functions detected!")
        print("Move import statements to the top of the file.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
