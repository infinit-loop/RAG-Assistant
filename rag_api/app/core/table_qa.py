"""Text-to-pandas analytical Q&A over a single DataFrame.

The LLM never computes numbers — it only writes a one-line pandas *expression*;
the math runs in pandas, so results are exact and reproducible. Designed for
small/medium tables (in-memory pandas is ample up to ~10k rows and beyond for
simple aggregations).

Pipeline (mirrors the standard Text-to-SQL flow, pandas instead of SQL):
  1. routing            -> decided upstream (agent classifies "structured")
  2. schema linking     -> build_catalog(): columns, dtypes, sample/known values
  3. generation         -> LLM emits ONE pandas expression over `df`, grounded
                           on the catalog + few-shot examples
  4. validation         -> AST whitelist: only `df`/`pd`, no dunders, no I/O,
                           no apply/eval/query; single read-only expression
  5. execution          -> restricted eval() on a copy of df (no builtins)
  6. self-correction    -> on validate/exec error, feed the error back, retry
  7. formatting         -> render the value; numbers come only from the engine
"""
from __future__ import annotations
import ast
import re
import pandas as pd

from app.common.logging import get_logger

log = get_logger("table_qa")

# ---------------------------------------------------------------------------
# 4. Validation — AST whitelist
# ---------------------------------------------------------------------------

# Node types the expression may contain. Anything else (imports, lambdas,
# comprehensions, assignments, attribute writes, f-strings, ...) is rejected.
_ALLOWED_NODES = tuple(
    t for t in (
        ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load,
        ast.Constant, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict,
        ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
        getattr(ast, "Index", None),  # py<3.9 only; None filtered out below
    ) if t is not None
)

# The only bare names allowed at the root of the expression.
_ALLOWED_NAMES = {"df", "pd"}
_SAFE_BUILTINS = {"len": len, "round": round, "min": min, "max": max,
                  "sum": sum, "abs": abs, "sorted": sorted,
                  "int": int, "float": float, "str": str, "bool": bool}
_ALLOWED_ROOT = _ALLOWED_NAMES | set(_SAFE_BUILTINS)

# Methods / accessors / properties the expression may call. Deliberately omits
# anything that runs arbitrary callables (apply, applymap, map, pipe, agg with
# a lambda is still limited to named reducers here, transform) or touches I/O
# (to_csv, to_pickle, read_*, eval, query).
_ALLOWED_ATTRS = {
    # selection / shape
    "loc", "iloc", "at", "iat", "columns", "index", "values", "dtypes",
    "shape", "size", "T", "to_frame", "reset_index", "set_index", "rename",
    "squeeze", "name", "empty",
    # grouping / aggregation
    "groupby", "agg", "aggregate", "sum", "mean", "median", "mode", "min",
    "max", "count", "std", "var", "quantile", "prod", "cumsum", "cumcount",
    "first", "last", "idxmax", "idxmin", "rank", "nlargest", "nsmallest",
    # ordering / slicing
    "sort_values", "sort_index", "head", "tail",
    # uniqueness / membership
    "value_counts", "unique", "nunique", "drop_duplicates", "isin", "between",
    # nulls / casting / cleaning
    "isna", "notna", "isnull", "notnull", "fillna", "dropna", "astype",
    "round", "abs", "clip", "where", "mask", "replace", "drop",
    # describe / stats
    "describe", "corr", "cov",
    # binary helpers
    "add", "sub", "mul", "div", "gt", "lt", "ge", "le", "eq", "ne",
    # reshaping
    "pivot_table", "melt", "merge", "stack", "unstack", "transpose",
    # constructors / converters on pd
    "Series", "DataFrame", "to_datetime", "to_numeric", "cut", "qcut",
    "Timestamp", "Timedelta", "date_range",
    # str / dt / cat accessors and their common methods
    "str", "dt", "cat",
    "contains", "startswith", "endswith", "lower", "upper", "strip", "split",
    "year", "month", "day", "hour", "date", "weekday", "dayofweek",
    "categories", "codes",
}


def validate(expr: str) -> ast.Expression:
    """Parse and whitelist-check a pandas expression. Raises ValueError if it
    contains anything outside the allowed surface."""
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")
    if "\n" in expr or ";" in expr:
        raise ValueError("expression must be a single line")
    try:
        tree = ast.parse(expr, mode="eval")  # mode="eval" forbids statements
    except SyntaxError as e:
        raise ValueError(f"syntax error: {e.msg}")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError(f"disallowed dunder/private attr: {node.attr}")
            if node.attr not in _ALLOWED_ATTRS:
                raise ValueError(f"attribute not allowed: '{node.attr}'")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_ROOT:
            raise ValueError(f"name not allowed: '{node.id}'")
    return tree


# ---------------------------------------------------------------------------
# 5. Execution — restricted eval on a copy
# ---------------------------------------------------------------------------
def execute(tree: ast.Expression, df: pd.DataFrame):
    """Evaluate a validated tree with no builtins and only df/pd in scope.
    Runs on a copy so any (blocked) mutation can't corrupt the stored frame."""
    code = compile(tree, "<table-qa>", "eval")
    globals_ = {"__builtins__": {}}
    locals_ = {"df": df.copy(), "pd": pd, **_SAFE_BUILTINS}
    return eval(code, globals_, locals_)  # noqa: S307 - sandboxed above


# ---------------------------------------------------------------------------
# 2. Schema linking — the catalog the LLM is grounded on
# ---------------------------------------------------------------------------
def build_catalog(df: pd.DataFrame, table_name: str, max_values: int = 15) -> str:
    """Describe the table so the model can write valid, value-linked pandas:
    column names, dtypes, and for low-cardinality text columns the actual
    distinct values (so 'the northwest branch' maps to the literal 'Northwest')."""
    lines = [f"DataFrame `df` loaded from '{table_name}', {len(df)} rows.",
             "Columns (name : dtype):"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        series = df[col]
        if series.dtype == object or str(series.dtype).startswith("category"):
            uniq = series.dropna().unique()
            if len(uniq) <= max_values:
                vals = ", ".join(map(str, uniq[:max_values]))
                lines.append(f"  - {col} : {dtype}  (values: {vals})")
            else:
                ex = ", ".join(map(str, uniq[:3]))
                lines.append(f"  - {col} : {dtype}  (e.g. {ex}; "
                             f"{len(uniq)} distinct)")
        else:
            try:
                lo, hi = series.min(), series.max()
                lines.append(f"  - {col} : {dtype}  (range {lo} .. {hi})")
            except Exception:
                lines.append(f"  - {col} : {dtype}")
    return "\n".join(lines)


# 3. Few-shot examples (patterns, not this table's columns — the model adapts
# column names from the catalog above).
_FEWSHOT = (
    "Examples (adapt the column names to the schema above):\n"
    "Q: Which category has the highest total amount?\n"
    "A: df.groupby('category')['amount'].sum().idxmax()\n"
    "Q: What is the average score?\n"
    "A: df['score'].mean()\n"
    "Q: Show the top 5 items by quantity.\n"
    "A: df.nlargest(5, 'quantity')[['item', 'quantity']]\n"
    "Q: How many rows have status over 100?\n"
    "A: int((df['status'] > 100).sum())\n"
)

_SYSTEM = (
    "You translate a question into ONE single-line pandas expression that "
    "computes the answer from a DataFrame named `df`.\n"
    "Rules:\n"
    "- Return ONLY the expression. No code fences, no `df =`, no print, no "
    "explanation.\n"
    "- Use only `df` and `pd`. Never import, read/write files, or use apply, "
    "map, eval, or query.\n"
    "- It must be read-only (no assignment, no inplace=True).\n"
    "- Use the exact column names and values from the provided schema."
)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_EXPR_HINT = re.compile(r"^(df|pd|int|float|len|round|sum|min|max|abs|sorted|bool|str)\b")


def _clean(raw: str) -> str:
    """Extract a single pandas expression from a model reply that may include
    reasoning (<think>…</think>), code fences, prose, or an 'A:' prefix.

    Strategy: drop think blocks and fences, then pick the most expression-like
    line (one that starts with df/pd/a safe builtin); fall back to the last
    non-empty line."""
    if not raw:
        return ""
    text = _THINK.sub("", raw)
    text = text.replace("```python", "```")
    text = text.replace("```", "\n")          # fences become line breaks
    lines = []
    for ln in text.splitlines():
        ln = ln.strip().strip("`").strip()
        for prefix in ("A:", "Answer:", "answer:", "Expression:", "python"):
            if ln.startswith(prefix):
                ln = ln[len(prefix):].strip()
        if ln:
            lines.append(ln)
    if not lines:
        return ""
    # Prefer a line that looks like a pandas expression (scan last-first so a
    # final answer wins over earlier scratch work).
    for ln in reversed(lines):
        if _EXPR_HINT.match(ln):
            return ln
    return lines[-1]


# ---------------------------------------------------------------------------
# 7. Formatting — render the engine's value (never the model's words)
# ---------------------------------------------------------------------------
def format_result(result, limit: int = 20) -> str:
    """Render a scalar / Series / DataFrame for display. Tables are capped at
    `limit` rows (the pandas equivalent of forcing a SQL LIMIT)."""
    if isinstance(result, pd.DataFrame):
        shown = result.head(limit)
        body = shown.to_string()
        more = (f"\n… ({len(result)} rows total, showing {limit})"
                if len(result) > limit else "")
        return f"```\n{body}{more}\n```"
    if isinstance(result, pd.Series):
        shown = result.head(limit)
        body = shown.to_string()
        more = (f"\n… ({len(result)} rows total, showing {limit})"
                if len(result) > limit else "")
        return f"```\n{body}{more}\n```"
    if isinstance(result, float):
        return f"**{result:,.2f}**"
    if hasattr(result, "item"):           # numpy scalar
        try:
            v = result.item()
            return f"**{v:,.2f}**" if isinstance(v, float) else f"**{v}**"
        except Exception:
            pass
    return f"**{result}**"


# ---------------------------------------------------------------------------
# Orchestrator (steps 3-7 with the 6. self-correction loop)
# ---------------------------------------------------------------------------
def answer_tabular(llm, question: str, df: pd.DataFrame, table_name: str,
                   retries: int = 2, row_limit: int = 20) -> dict:
    """Generate -> validate -> execute -> (retry) -> format. Returns a dict
    with ok/answer/expr, or ok=False/error if every attempt failed."""
    catalog = build_catalog(df, table_name)
    user = (f"{catalog}\n\n{_FEWSHOT}\n"
            f"Now answer this question with one pandas expression.\n"
            f"Q: {question}\nA:")
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]

    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = llm.generate_code(messages)
        except Exception as e:  # network / API failure
            log.warning("text2pandas: LLM call failed: %s", e)
            return {"ok": False, "error": f"LLM call failed: {e}", "expr": None}
        expr = _clean(raw)
        log.info("text2pandas: attempt %d expr=%s", attempt + 1, expr)
        try:
            tree = validate(expr)
            result = execute(tree, df)
        except Exception as e:  # validation or execution error -> self-correct
            last_err = f"{type(e).__name__}: {e}"
            log.warning("text2pandas: rejected (%s) -> self-correcting", last_err)
            messages.append({"role": "assistant", "content": expr})
            messages.append({"role": "user",
                             "content": (f"That expression failed with: {last_err}. "
                                         "Return a corrected single-line pandas "
                                         "expression. Only the expression.")})
            continue
        log.info("text2pandas: OK expr=%s", expr)
        return {"ok": True, "expr": expr,
                "answer": format_result(result, row_limit)}
    log.warning("text2pandas: gave up after %d attempts (%s)",
                retries + 1, last_err)
    return {"ok": False, "error": last_err, "expr": None}
