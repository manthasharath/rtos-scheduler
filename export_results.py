"""
export_results.py
-----------------
CSV and Excel (OpenPyXL) export helpers.
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write several DataFrames into one .xlsx workbook (one sheet each)."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or len(df) == 0:
                continue
            sheet = name[:31]  # Excel sheet-name limit
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, width + 2), 45)
    return buf.getvalue()
