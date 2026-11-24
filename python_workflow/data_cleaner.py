"""Automated Financial Data Cleaning & Visualization Diagnostics Module.

Handles heterogeneous messy datasets:
- Deduplication & schema normalization
- Sector-median missing value imputation
- IQR / Winsorization outlier capping
- Automated multi-panel distribution & optimization surface chart generation
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"


def create_sample_messy_panel(seed: int = 202608) -> Path:
    """Create a realistic messy multi-sector financial indicator panel with missing values & outliers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sectors = ["Commercial Bank", "Securities", "New Energy", "Semiconductor", "Nonferrous Metals"]

    rows = []
    for i in range(1, 61):
        sec = sectors[i % len(sectors)]
        base_roe = {"Commercial Bank": 11.2, "Securities": 8.4, "New Energy": 14.5, "Semiconductor": 12.1, "Nonferrous Metals": 13.0}[sec]
        base_pe = {"Commercial Bank": 6.2, "Securities": 18.5, "New Energy": 26.0, "Semiconductor": 38.0, "Nonferrous Metals": 15.2}[sec]
        roe = base_roe + rng.normal(0, 2.8)
        pe = max(3.0, base_pe + rng.normal(0, 5.5))
        rev_yoy = rng.normal(12.5, 9.0)
        debt_ratio = np.clip(rng.normal(62.0 if sec == "Commercial Bank" else 48.0, 11.0), 15.0, 92.0)

        rows.append(
            {
                "ticker": f"FIN{600000 + i}",
                "sector": sec,
                "roe_pct": round(roe, 2),
                "pe_ttm": round(pe, 2),
                "revenue_yoy_pct": round(rev_yoy, 2),
                "debt_to_asset_pct": round(debt_ratio, 2),
            }
        )

    df = pd.DataFrame(rows)
    # Inject realistic data quality issues: missing cells, extreme outliers, duplicate rows
    df.loc[[4, 17, 33], "roe_pct"] = np.nan
    df.loc[[9, 25], "pe_ttm"] = np.nan
    df.loc[[12, 44], "revenue_yoy_pct"] = np.nan
    df.loc[7, "pe_ttm"] = 480.0         # extreme valuation outlier
    df.loc[28, "revenue_yoy_pct"] = 245.0  # one-off M&A outlier
    df.loc[51, "roe_pct"] = -65.0      # distressed outlier
    df = pd.concat([df, df.iloc[[2, 15, 30]]], ignore_index=True)  # duplicates

    raw_path = DATA_DIR / "raw_financial_panel.csv"
    df.to_csv(raw_path, index=False)
    return raw_path


def clean_financial_panel(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    """Run automated deduplication, sector-median imputation, and 2%/98% Winsorization."""
    raw = pd.read_csv(raw_path)
    n_raw = len(raw)
    missing_before = int(raw.isna().sum().sum())

    df = raw.drop_duplicates(subset=["ticker"], keep="first").copy()
    n_dedup_removed = n_raw - len(df)

    num_cols = ["roe_pct", "pe_ttm", "revenue_yoy_pct", "debt_to_asset_pct"]
    # Sector-median imputation
    for col in num_cols:
        df[col] = df[col].fillna(df.groupby("sector")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # Winsorize at [2%, 98%] to neutralize extreme reporting anomalies
    outliers_capped = 0
    for col in num_cols:
        lo, hi = df[col].quantile([0.02, 0.98])
        capped_mask = (df[col] < lo) | (df[col] > hi)
        outliers_capped += int(capped_mask.sum())
        df[col] = df[col].clip(lower=lo, upper=hi).round(2)

    # Compute standardized composite quality score
    z_roe = (df["roe_pct"] - df["roe_pct"].mean()) / df["roe_pct"].std()
    z_growth = (df["revenue_yoy_pct"] - df["revenue_yoy_pct"].mean()) / df["revenue_yoy_pct"].std()
    z_val = -(df["pe_ttm"] - df["pe_ttm"].mean()) / df["pe_ttm"].std()
    df["composite_score"] = (0.45 * z_roe + 0.35 * z_growth + 0.20 * z_val).round(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = OUTPUT_DIR / "cleaned_financial_panel.csv"
    df.to_csv(clean_path, index=False)

    audit = {
        "rows_raw": n_raw,
        "duplicates_removed": n_dedup_removed,
        "missing_cells_imputed": missing_before,
        "outliers_winsorized": outliers_capped,
        "rows_clean": len(df),
        "clean_csv": str(clean_path),
    }
    return df, audit


def generate_workflow_chart(clean_df: pd.DataFrame) -> Path:
    """Render a 2-panel automated analytics chart: (1) 2D optimization contour, (2) cleaned sector ROE vs PE."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left panel: 2D Constrained Optimization Response Surface (mirroring the Web Simulator)
    x = np.linspace(0, 80, 120)
    y = np.linspace(0, 95, 120)
    X, Y = np.meshgrid(x, y)
    budget, lam, rho, c1, c2 = 100.0, 1.8, 0.25, 1.2, 1.0
    Z = 0.13 * X + 0.085 * Y - 0.5 * lam * (0.0022 * X**2 + 0.0014 * Y**2 + 2 * rho * 0.00175 * X * Y) - 0.015 * (X + Y)
    Z_masked = np.ma.masked_where(c1 * X + c2 * Y > budget, Z)

    cf = ax1.contourf(X, Y, Z_masked, levels=18, cmap="viridis")
    fig.colorbar(cf, ax=ax1, label="Utility f(x, y)")
    x_line = np.linspace(0, budget / c1, 50)
    ax1.plot(x_line, (budget - c1 * x_line) / c2, color="#e74c3c", ls="--", lw=2.0, label="Budget Constraint")

    # Find discrete optimum with step = 2
    best_pt = None
    for xi in range(0, 84, 2):
        for yi in range(0, 101, 2):
            if c1 * xi + c2 * yi <= budget:
                val = 0.13 * xi + 0.085 * yi - 0.5 * lam * (0.0022 * xi**2 + 0.0014 * yi**2 + 2 * rho * 0.00175 * xi * yi) - 0.015 * (xi + yi)
                if best_pt is None or val > best_pt[2]:
                    best_pt = (xi, yi, val)
    ax1.scatter([best_pt[0]], [best_pt[1]], s=130, color="#f39c12", edgecolor="white", lw=1.8, zorder=5,
                label=f"Discrete Opt ({best_pt[0]}, {best_pt[1]}) = {best_pt[2]:.2f}")
    ax1.set_title("2D Constrained Utility Surface & Discrete Optimum", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Asset X Discrete Lots")
    ax1.set_ylabel("Asset Y Discrete Lots")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Right panel: Cleaned Sector Scatter (PE vs ROE sized by Revenue YoY)
    palette = ["#1f4e79", "#27ae60", "#e67e22", "#8e44ad", "#c0392b"]
    for idx, (sec, grp) in enumerate(clean_df.groupby("sector")):
        ax2.scatter(
            grp["pe_ttm"],
            grp["roe_pct"],
            s=np.clip(grp["revenue_yoy_pct"] + 20, 12, 50) * 2.5,
            alpha=0.75,
            color=palette[idx % len(palette)],
            label=sec,
        )
    ax2.set_title("Cleaned Multi-Sector Panel: Valuation (PE) vs ROE", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("PE (TTM, Winsorized)")
    ax2.set_ylabel("ROE (%)")
    ax2.legend(loc="lower right", fontsize=8.2)

    plt.tight_layout()
    chart_path = ASSETS_DIR / "workflow_summary_chart.png"
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path
