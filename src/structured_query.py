import re

AGG_KEYWORDS = {
    "highest": "max", "maximum": "max", "most expensive": "max", "largest": "max",
    "lowest": "min", "minimum": "min", "cheapest": "min", "smallest": "min",
    "average": "mean", "mean": "mean",
    "total": "sum", "sum": "sum",
    "how many": "count", "count": "count", "number of": "count",
}

ID_PATTERN = re.compile(r"\b(?:id|number|no\.?)\s*[:#]?\s*(\d+)\b", re.IGNORECASE)


def find_id_column(df):
    for col in df.columns:
        if re.search(r"\bid\b", str(col), re.IGNORECASE):
            return col
    return None


def try_exact_lookup(query, tables):
    match = ID_PATTERN.search(query)
    if not match:
        return None
    target_id = int(match.group(1))

    for table_name, df in (tables or {}).items():
        if df is None:
            continue
        id_col = find_id_column(df)
        if id_col is None:
            continue
        try:
            row = df[df[id_col].astype(str) == str(target_id)]
        except Exception:
            continue
        if not row.empty:
            r = row.iloc[0]
            details = "\n".join(f"{c}: {r[c]}" for c in df.columns)
            return f"From {table_name} (row with {id_col} = {target_id}):\n{details}"
    return None


def try_aggregation(query, tables):
    q = query.lower()
    op = next((v for k, v in AGG_KEYWORDS.items() if k in q), None)
    if op is None:
        return None

    results = []
    for table_name, df in (tables or {}).items():
        if df is None:
            continue
        numeric_cols = df.select_dtypes(include="number").columns
        target_cols = [c for c in numeric_cols if str(c).lower() in q] or list(numeric_cols)

        for col in target_cols:
            try:
                if op == "max":
                    idx = df[col].idxmax()
                    results.append(f"In {table_name}, the row with the highest {col} ({df.loc[idx, col]}) is:\n" +
                                    "\n".join(f"{c}: {df.loc[idx, c]}" for c in df.columns))
                elif op == "min":
                    idx = df[col].idxmin()
                    results.append(f"In {table_name}, the row with the lowest {col} ({df.loc[idx, col]}) is:\n" +
                                    "\n".join(f"{c}: {df.loc[idx, c]}" for c in df.columns))
                elif op == "mean":
                    results.append(f"In {table_name}, the average {col} is {df[col].mean():.2f}.")
                elif op == "sum":
                    results.append(f"In {table_name}, the total {col} is {df[col].sum():.2f}.")
                elif op == "count":
                    results.append(f"{table_name} has {len(df)} rows.")
            except Exception:
                continue
    return "\n\n".join(results) if results else None