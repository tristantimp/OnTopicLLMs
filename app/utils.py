# utils.py
import os
import csv
import json
from typing import Any, List, Dict

import pandas as pd


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase & underscore column names so we're robust to variations.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df


def parse_conversation_any(value: Any) -> Any:
    """
    Attempt to parse a conversation field into a list[dict] with keys {'role','content'}.
    Accepts:
      - already a list[dict]
      - JSON string
      - Python-literal-like string (single quotes) -> try json.loads after replace
      - otherwise return raw value
    """
    # already structured?
    if isinstance(value, list) and all(isinstance(x, dict) for x in value):
        return value

    # strings
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            # if it's a dict, wrap
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        # try Python-ish to JSON (single quotes -> double quotes) as a best-effort
        if ("'" in s) and ('"' not in s):
            try:
                candidate = s.replace("'", '"')
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                pass

        # give up: return raw
        return value

    # anything else: return as-is
    return value


def ensure_data_dir(data_dir: str):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def safe_append_row(path: str, row: dict, header_columns: List[str]):
    """
    Append a row to CSV creating the file with header if it doesn't exist.
    If the existing file's header is missing required columns, transparently
    upgrades the file by rewriting the header and backfilling missing fields.
    """
    exists = os.path.exists(path)

    # Ensure payload has all declared columns
    payload = {col: row.get(col, "") for col in header_columns}

    if exists:
        # Read current header
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                current_header = next(reader)
            except StopIteration:
                current_header = []

        # If header mismatch (missing columns or different order), migrate
        if current_header != header_columns:
            # Read all rows with DictReader using the old header
            with open(path, "r", encoding="utf-8", newline="") as f_in:
                old_reader = csv.DictReader(f_in)
                old_rows = list(old_reader)

            # Rewrite file with new header, map old rows to new schema
            with open(path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=header_columns)
                writer.writeheader()
                for old in old_rows:
                    mapped = {col: old.get(col, "") for col in header_columns}
                    writer.writerow(mapped)
            # After migration, proceed to append the new row below

    # Append (create if needed)
    file_exists_after_maybe_migrate = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_columns)
        if not file_exists_after_maybe_migrate:
            writer.writeheader()
        writer.writerow(payload)
