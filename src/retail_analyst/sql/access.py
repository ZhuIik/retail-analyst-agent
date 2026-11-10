"""Explicit refusal for literals outside the user's access scope.

This is a convenience for an honest answer, not protection: the scoped tables already hide
other brands, so without this check the model would see 0 rows and claim "X had no sales".
It only covers the obvious forms (`col = 'x'`, `col IN ('x', ...)`, `col LIKE 'x%'`,
optionally wrapped in LOWER/UPPER/TRIM); completeness is not promised.
"""

from __future__ import annotations

import re

from sqlglot import exp

from retail_analyst.contracts import AccessScope

_CASE_WRAPPERS = (exp.Lower, exp.Upper, exp.Trim)


def find_foreign_literal(root: exp.Expression, access: AccessScope) -> str | None:
    """Return the first literal compared to the access column that the user may not see."""
    if access.is_unrestricted:
        return None
    column = access.column
    allowed = {v.casefold() for v in access.values}

    for node in root.find_all(exp.EQ, exp.In, exp.Like):
        if _is_negated(node):  # `brand NOT LIKE 'X%'` excludes X, it does not ask for it
            continue
        if isinstance(node, exp.EQ):
            for col_side, lit_side in ((node.this, node.expression), (node.expression, node.this)):
                if _is_column(col_side, column) and _is_string(lit_side) and lit_side.this.casefold() not in allowed:
                    return lit_side.this
        elif isinstance(node, exp.In):
            if _is_column(node.this, column):
                for lit in node.expressions:
                    if _is_string(lit) and lit.this.casefold() not in allowed:
                        return lit.this
        elif _is_column(node.this, column) and _is_string(node.expression):
            pattern = node.expression.this
            if not any(_like_matches(pattern, value) for value in access.values):
                return pattern
    return None


def foreign_literal_message(access: AccessScope, value: str) -> str:
    own = ", ".join(f"'{v}'" for v in access.values) or "none"
    return f"You don't have access to {access.column} '{value}'. Your access covers {access.column}: {own}."


def _is_negated(node: exp.Expression) -> bool:
    if node.args.get("negate"):  # sqlglot keeps `NOT LIKE` as Like(negate=True)
        return True
    parent = node.parent
    while isinstance(parent, exp.Paren):
        parent = parent.parent
    return isinstance(parent, exp.Not)


def _is_column(node: exp.Expression, column: str | None) -> bool:
    while isinstance(node, _CASE_WRAPPERS):
        node = node.this
    return isinstance(node, exp.Column) and node.name.lower() == column


def _is_string(node: exp.Expression) -> bool:
    return isinstance(node, exp.Literal) and node.is_string


def _like_matches(pattern: str, value: str) -> bool:
    """SQL LIKE semantics (`%`, `_`, backslash escape), case-insensitive like the other checks."""
    regex, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            regex.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        regex.append(".*" if ch == "%" else "." if ch == "_" else re.escape(ch))
        i += 1
    return re.fullmatch("".join(regex), value, flags=re.IGNORECASE | re.DOTALL) is not None
