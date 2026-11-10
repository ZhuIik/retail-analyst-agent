"""Demographics in aggregates only, with a minimum group size (k-anonymity; D10, D11).

Client requirement: city, state, country, age, gender and traffic source are allowed in
aggregates only, groups below `sql.min_group_size` customers are hidden, pseudonymous ids for
individuals are fine. These columns are quasi-identifiers: "the one 34-year-old customer in
Wyoming" identifies a person even without a name.

Split of work:
- the guard (here) only *requires* that a query touching a demographic column outputs
  `customers = COUNT(DISTINCT <customer pseudonym>)` and does not list customer pseudonyms
  next to demographics;
- run_sql (tools/data_tools.py) then drops every result row whose `customers` is NULL or below
  the threshold, before anything (model, state, saved reports) sees it.
Suppression on the result is the protection; rule (b) mostly gives the model a clearer
message, since a per-customer row has customers = 1 and would be hidden anyway.

The lineage check is a deliberately small, conservative heuristic: it follows plain column
pass-throughs (aliases, CTEs, subqueries, `SELECT *` over one derived source) and rejects what
it cannot resolve. It is not a full proof of the query's semantics; production would rely on
BigQuery aggregation-threshold analysis rules on authorized views instead.
"""

from __future__ import annotations

from typing import Any

from sqlglot import exp

from retail_analyst.contracts import ErrorKind
from retail_analyst.sql.rules import GuardViolation

GROUP_SIZE_COLUMN = "customers"
_MAX_DEPTH = 10  # CTE chains deeper than this are not followed (rejected, fail closed)

# What a resolved column is: a base-table column (table, column) or the defining expression.
Origin = tuple[str, str] | exp.Expression | None


def demographic_columns(schema: dict) -> set[str]:
    """Names of `demographic: true` columns. Matched by name anywhere, like the PII check:
    `orders.gender` holds the same customer attribute as `users.gender`, so it is covered too."""
    return {
        col.lower()
        for table in schema.values()
        for col, spec in table["columns"].items()
        if (spec or {}).get("demographic")
    }


def check_demographics(
    root: exp.Query, refs: list[tuple[exp.Table, str]], schema: dict, min_group_size: int
) -> str | None:
    """Return the group-size output column if the query uses demographics; raise if it breaks the rule."""
    demographic = demographic_columns(schema)
    used = sorted({c.name.lower() for c in root.find_all(exp.Column) if c.name.lower() in demographic})
    if not used:
        return None

    lineage = _Lineage(root, refs, schema)
    branches = _branches(root)

    def violation(lead: str, fix: str) -> GuardViolation:
        return GuardViolation(
            ErrorKind.RULE_VIOLATION,
            f"{lead} ({', '.join(used)}). Demographics are allowed only in aggregates: {fix} "
            f"Groups with fewer than {min_group_size} customers are hidden.",
        )

    if any(lineage.depends_on_pseudonym(p) for select in branches for p in lineage.outputs(select)):
        raise violation(
            "This query outputs customer ids together with customer demographics",
            "remove customer ids from the final SELECT (counting them is fine), group by the demographic "
            "and add COUNT(DISTINCT u.id) AS customers.",
        )

    counts = [
        next((p for p in lineage.outputs(select) if p.alias_or_name.lower() == GROUP_SIZE_COLUMN), None)
        for select in branches
    ]
    if not counts or not all(c is not None and lineage.is_customer_count(c) for c in counts):
        raise violation(
            "This query uses customer demographics",
            "add COUNT(DISTINCT u.id) AS customers to the final SELECT "
            "(COUNT(DISTINCT user_id) also works; COUNT(*) does not).",
        )
    return counts[0].alias_or_name  # the output name BigQuery uses (first branch names a UNION)


def drop_small_groups(
    columns: list[str], rows: list[list[Any]], group_size_column: str, min_group_size: int
) -> tuple[list[list[Any]], int]:
    """Rows whose group-size value is >= min_group_size, and how many were dropped.

    Fails closed: a missing column, NULL or non-numeric value hides the row.
    """
    index = next((i for i, c in enumerate(columns) if c.lower() == group_size_column.lower()), None)
    if index is None:
        return [], len(rows)

    def big_enough(value: Any) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool) and value >= min_group_size

    kept = [row for row in rows if big_enough(row[index])]
    return kept, len(rows) - len(kept)


def _branches(root: exp.Expression) -> list[exp.Select]:
    """The SELECTs that produce the final output (several for UNION / INTERSECT / EXCEPT)."""
    while isinstance(root, exp.Subquery):
        root = root.this
    if isinstance(root, exp.SetOperation):
        return _branches(root.left) + _branches(root.right)
    return [root] if isinstance(root, exp.Select) else []


def _sources(select: exp.Select) -> list[exp.Expression]:
    from_ = select.args.get("from_") or select.args.get("from")
    return ([from_.this] if from_ else []) + [join.this for join in select.args.get("joins") or []]


def _source_alias(source: exp.Expression) -> str:
    return source.alias_or_name.lower()


class _Lineage:
    """Follows a column back through the model's CTEs and subqueries to where it is defined."""

    def __init__(self, root: exp.Expression, refs: list[tuple[exp.Table, str]], schema: dict) -> None:
        self.base = {id(table): name for table, name in refs}
        self.schema = schema
        self.ctes: dict[str, exp.Expression | None] = {}
        for cte in root.find_all(exp.CTE):
            name = cte.alias_or_name.lower()
            # Two CTEs with one name (nested WITHs): ambiguous, do not guess which one.
            self.ctes[name] = None if name in self.ctes else cte.this
        self.hashed = {
            (t, c.lower()) for t, spec in schema.items() for c, s in spec["columns"].items() if (s or {}).get("hashed")
        }

    # ---- outputs -----------------------------------------------------------------------

    def outputs(self, select: exp.Select, depth: int = 0) -> list[exp.Expression]:
        """Output expressions of a SELECT; `SELECT *` over one CTE/subquery expands to its outputs."""
        result: list[exp.Expression] = []
        for projection in select.expressions:
            star = projection.this if isinstance(projection, exp.Column) else projection
            if not isinstance(star, exp.Star):  # `*` and `t.*` only (COUNT(*) is a Count, not a Star)
                result.append(projection)
                continue
            sources = _sources(select)
            inner = self._query(sources[0]) if len(sources) == 1 else None
            if isinstance(inner, exp.Select) and depth < _MAX_DEPTH and not star.args.get("replace"):
                excluded = {e.name.lower() for e in star.args.get("except_") or star.args.get("except") or []}
                result += [p for p in self.outputs(inner, depth + 1) if p.alias_or_name.lower() not in excluded]
            else:
                result.append(projection)  # unexpandable: cannot be `customers`, so rule (a) fails closed
        return result

    # ---- rule (a): customers = COUNT(DISTINCT pseudonym) -------------------------------

    def is_customer_count(self, expr: exp.Expression, depth: int = 0) -> bool:
        node = expr.unalias()
        if isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct):
            args = node.this.expressions
            return len(args) == 1 and isinstance(args[0], exp.Column) and self.is_pseudonym(args[0])
        if isinstance(node, exp.Column) and depth < _MAX_DEPTH:
            origin = self.resolve(node)
            return isinstance(origin, exp.Expression) and self.is_customer_count(origin, depth + 1)
        return False

    # ---- rule (b): no customer pseudonyms next to demographics -------------------------

    def depends_on_pseudonym(self, expr: exp.Expression, depth: int = 0) -> bool:
        """True if the output carries customer pseudonyms other than as a COUNT argument
        (a plain id, CONCAT(id, ...), ARRAY_AGG(id), ...), followed through CTEs."""
        if depth >= _MAX_DEPTH:
            return False
        for column in expr.find_all(exp.Column):
            if _inside_count(column, stop=expr):
                continue
            origin = self.resolve(column)
            if isinstance(origin, tuple) and origin in self.hashed:
                return True
            if isinstance(origin, exp.Expression) and self.depends_on_pseudonym(origin, depth + 1):
                return True
        return False

    def is_pseudonym(self, column: exp.Column, depth: int = 0) -> bool:
        origin = self.resolve(column)
        if isinstance(origin, exp.Column) and depth < _MAX_DEPTH:
            return self.is_pseudonym(origin, depth + 1)
        return isinstance(origin, tuple) and origin in self.hashed

    # ---- resolution ----------------------------------------------------------------------

    def resolve(self, column: exp.Column) -> Origin:
        """Where `column` comes from in its own SELECT; None if unknown or ambiguous."""
        select = column.find_ancestor(exp.Select)
        if select is None:
            return None
        name, qualifier = column.name.lower(), column.table.lower()
        sources = _sources(select)
        if qualifier:
            candidates = [s for s in sources if _source_alias(s) == qualifier]
        else:
            candidates = [s for s in sources if self._provides(s, name)]
        if len(candidates) != 1:
            return None
        source = candidates[0]
        if (table := self.base.get(id(source))) is not None:
            return (table, name)
        query = self._query(source)
        projection = self._projection(query, name) if isinstance(query, exp.Select) else None
        return projection.unalias() if projection is not None else None

    def _provides(self, source: exp.Expression, name: str) -> bool:
        if (table := self.base.get(id(source))) is not None:
            return name in {c.lower() for c in self.schema[table]["columns"]}
        query = self._query(source)
        return isinstance(query, exp.Select) and self._projection(query, name) is not None

    def _query(self, source: exp.Expression) -> exp.Expression | None:
        if isinstance(source, exp.Subquery):
            return source.this
        if isinstance(source, exp.Table) and id(source) not in self.base and len(source.parts) == 1:
            return self.ctes.get(source.name.lower())
        return None

    def _projection(self, select: exp.Select, name: str) -> exp.Expression | None:
        return next((p for p in self.outputs(select) if p.alias_or_name.lower() == name), None)


def _inside_count(node: exp.Expression, stop: exp.Expression) -> bool:
    """True if `node` sits inside a COUNT(...) within `stop` (counting ids reveals no one)."""
    while node is not stop and node.parent is not None:
        node = node.parent
        if isinstance(node, exp.Count):
            return True
    return False
