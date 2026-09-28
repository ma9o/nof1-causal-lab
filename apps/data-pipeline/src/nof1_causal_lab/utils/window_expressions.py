"""The scalar window grammar shared by extraction and observation semantics."""

from __future__ import annotations

import ast


def window_summary_operator(expression: str) -> str:
    """Identify one modeled summary, allowing unit conversion and missingness guards.

    Row transforms belong inside the reduction. A window-wide Boolean, ratio of
    summaries, or nonlinear transform of a summary needs its own observation law;
    it cannot claim the semantics of a point reading or an interval mean.
    """
    root = ast.parse(expression, mode="eval").body
    reductions = {
        "first": "first",
        "last": "last",
        "sum": "sum",
        "mean": "mean",
        "std": "std",
        "count_true": "count",
        "count_non_null": "count",
    }
    window_functions = {*reductions, "any", "all", "min", "max"}

    def _scalar(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in window_functions:
                return True
            return all(_scalar(arg) for arg in node.args)
        return all(_scalar(child) for child in ast.iter_child_nodes(node))

    def _number(node: ast.AST) -> bool:
        return (isinstance(node, ast.Constant) and type(node.value) in (int, float)) or (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and _number(node.operand)
        )

    def _summary(node: ast.AST) -> str:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            operator = reductions.get(node.func.id)
            if operator is not None and len(node.args) == 1:
                nested = any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id in window_functions
                    for child in ast.walk(node.args[0])
                )
                if not nested:
                    return operator
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _summary(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            if _number(node.right):
                return _summary(node.left)
            if _number(node.left) and not isinstance(node.op, ast.Div):
                return _summary(node.right)
        if isinstance(node, ast.IfExp) and _scalar(node.test):
            if isinstance(node.body, ast.Constant) and node.body.value is None:
                return _summary(node.orelse)
            if isinstance(node.orelse, ast.Constant) and node.orelse.value is None:
                return _summary(node.body)
        raise ValueError(
            "computed_rule requires one supported window summary (first, last, sum, mean, std, "
            "count_true, or count_non_null), optionally with unit conversion or a missingness "
            "guard. Put row transforms inside the summary; other window formulas have no "
            "supported observation semantics."
        )

    return _summary(root)
