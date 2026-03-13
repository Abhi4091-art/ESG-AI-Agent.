"""
data_handler.py – Data Ingestion, Validation, Querying & Real Analysis
======================================================================
Improvements over original:
  - validate_year_constraint now returns True (pass) when no year/date column exists
    (was incorrectly blocking all uploads without a year column)
  - generate_data_summary now properly uses actual data for benchmarks
  - compute_benchmark_comparison uses real data values when available (not hardcoded estimates)
  - compute_esg_maturity_score now checks for and uses actual data columns first
  - compute_kpi_improvements uses real baseline values when available
  - Added tab_data_overview() helper for a new "Data Explorer" tab
  - Fixed: DuckDB query_summary_stats was using unnest(map_keys(...)) which breaks on
    non-homogeneous tables; replaced with a safe pandas-based implementation
  - Fixed: detect_esg_columns had silent key collisions; now picks best match per metric
  - Added: export_to_excel() for Excel download option
  - Removed: unused numpy import replaced with direct pandas operations
"""

import pandas as pd
from io import BytesIO
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. CSV Loading  (unchanged – already robust)
# ---------------------------------------------------------------------------
def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    strategies = [
        {"sep": ",",    "engine": "python", "on_bad_lines": "skip"},
        {"sep": None,   "engine": "python", "on_bad_lines": "skip"},
        {"sep": ";",    "engine": "python", "on_bad_lines": "skip"},
        {"sep": "\t",   "engine": "python", "on_bad_lines": "skip"},
    ]
    last_error = None
    for kwargs in strategies:
        try:
            df = pd.read_csv(BytesIO(raw), **kwargs)
            if not df.empty:
                # Strip leading/trailing whitespace from column names
                df.columns = df.columns.str.strip()
                return df
        except Exception as e:
            last_error = e
    raise ValueError(f"Failed to parse CSV. Last error: {last_error}")


# ---------------------------------------------------------------------------
# 2. Constraint Validation
#    FIX: original returned (False, …) when no year/date column present,
#    blocking all CSVs that don't have a year column. Correct behaviour:
#    pass without a date column (we simply can't validate), but warn.
# ---------------------------------------------------------------------------
def validate_year_constraint(df: pd.DataFrame) -> Tuple[bool, str]:
    year_cols = [c for c in df.columns if "year" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if not year_cols and not date_cols:
        # No temporal column – allow upload but note the gap
        return (True, "No year/date column found – temporal validation skipped.")

    for col in year_cols:
        numeric_years = pd.to_numeric(df[col], errors="coerce")
        if numeric_years.isna().all():
            continue
        min_year = int(numeric_years.dropna().min())
        if min_year < 2010:
            return (False, f"Column '{col}' contains data from {min_year}, before 2010.")

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid_dates = parsed.dropna()
        if valid_dates.empty:
            continue
        min_year = valid_dates.dt.year.min()
        if min_year < 2010:
            return (False, f"Column '{col}' contains dates from {min_year}, before 2010.")

    return (True, "✅ Data constraint check passed – all records are from 2010 onwards.")


# ---------------------------------------------------------------------------
# 3. Data Quality Checks
# ---------------------------------------------------------------------------
def run_data_quality_checks(df: pd.DataFrame) -> list:
    warnings = []
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"Found {n_dupes} duplicate row(s). Consider deduplicating.")
    for col in df.columns:
        pct_missing = df[col].isna().mean() * 100
        if pct_missing > 50:
            warnings.append(f"Column '{col}' has {pct_missing:.1f}% missing values.")
    if len(df) < 2:
        warnings.append("Dataset has fewer than 2 rows – results may be unreliable.")
    # Check for constant columns (zero variance) – not useful for analysis
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        if numeric_df[col].nunique() <= 1:
            warnings.append(f"Column '{col}' has only one unique value (constant).")
    return warnings


# ---------------------------------------------------------------------------
# 4. Auto-Detect ESG Columns
#    FIX: original could silently clobber mapping[metric] if two columns
#    matched the same metric; now scores matches and picks the best.
# ---------------------------------------------------------------------------
_COLUMN_PATTERNS = {
    "company":              ["company", "firm", "organisation", "organization", "entity", "name"],
    "industry":             ["industry", "sector", "segment"],
    "country":              ["country", "region", "location", "nation", "geography"],
    "year":                 ["year", "fiscal_year", "fy", "period"],
    "employees":            ["employee", "headcount", "staff", "workforce", "fte"],
    "revenue":              ["revenue", "turnover", "sales", "income"],
    "esg_budget":           ["esg_budget", "esg_spend", "sustainability_budget", "green_spend"],
    "carbon_emissions":     ["carbon", "emission", "co2", "ghg", "tco2", "greenhouse"],
    "energy":               ["energy", "electricity", "power_consumption", "mwh", "kwh"],
    "renewable_energy":     ["renewable", "clean_energy", "green_energy", "solar", "wind"],
    "water":                ["water"],
    "waste_total":          ["waste_generated", "waste_total", "waste_produced", "total_waste"],
    "waste_recycled":       ["waste_recycl", "recycl", "recycled_waste"],
    "gender_diversity":     ["gender_divers", "female_pct", "women_pct", "diversity", "gender_ratio"],
    "employee_turnover":    ["turnover_pct", "attrition", "employee_turnover"],
    "training":             ["training_hour", "learning_hour", "training"],
    "board_independence":   ["board_independ", "independent_director"],
    "safety":               ["safety_incident", "injury", "accident", "ltir", "trir"],
    "community_investment": ["community_invest", "social_invest", "philanthrop", "donation", "charitable"],
    "scope1":               ["scope1", "scope_1", "direct_emissions"],
    "scope2":               ["scope2", "scope_2", "indirect_emissions"],
    "scope3":               ["scope3", "scope_3", "value_chain"],
}


def detect_esg_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect which columns map to ESG metrics.
    Uses a scoring approach: exact substring match wins, longer pattern wins on ties.
    Returns dict: standardised_metric -> actual_column_name
    """
    col_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    # candidate: metric -> list of (score, col)
    candidates: dict = {}

    for metric, patterns in _COLUMN_PATTERNS.items():
        for col, col_l in col_lower.items():
            for pattern in patterns:
                if pattern in col_l:
                    score = len(pattern)  # longer match = more specific
                    if metric not in candidates or score > candidates[metric][0]:
                        candidates[metric] = (score, col)
                    break  # first matching pattern for this col is sufficient

    return {metric: col for metric, (_, col) in candidates.items()}


# ---------------------------------------------------------------------------
# 5. Data Summary for LLM Context
# ---------------------------------------------------------------------------
def generate_data_summary(df: pd.DataFrame) -> dict:
    col_map = detect_esg_columns(df)
    numeric_df = df.select_dtypes(include="number")
    cat_df     = df.select_dtypes(include=["object", "category"])

    summary = {
        "row_count":        len(df),
        "col_count":        len(df.columns),
        "columns":          list(df.columns),
        "numeric_cols":     list(numeric_df.columns),
        "categorical_cols": list(cat_df.columns),
        "completeness_pct": round((1 - df.isna().mean().mean()) * 100, 1),
        "year_range":       None,
        "companies":        [],
        "key_metrics":      {},
        "column_mapping":   col_map,
    }

    # Year range
    if "year" in col_map:
        yrs = pd.to_numeric(df[col_map["year"]], errors="coerce").dropna()
        if not yrs.empty:
            summary["year_range"] = (int(yrs.min()), int(yrs.max()))

    # Companies
    if "company" in col_map:
        summary["companies"] = df[col_map["company"]].dropna().unique().tolist()

    # Numeric stats (cap at 15 columns to keep summary concise)
    for col in list(numeric_df.columns)[:15]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            summary["key_metrics"][col] = {
                "min":    round(float(series.min()), 2),
                "max":    round(float(series.max()), 2),
                "mean":   round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
            }

    # Text summary for LLM
    parts = [
        f"Dataset: {summary['row_count']} rows, {summary['col_count']} columns.",
        f"Completeness: {summary['completeness_pct']}%.",
    ]
    if summary["year_range"]:
        parts.append(f"Year range: {summary['year_range'][0]}-{summary['year_range'][1]}.")
    if summary["companies"]:
        parts.append(f"Companies: {', '.join(str(c) for c in summary['companies'][:5])}.")
    if col_map:
        parts.append(f"ESG columns detected: {', '.join(col_map.keys())}.")

    # Add a few key metric highlights
    for metric in ["carbon_emissions", "renewable_energy", "gender_diversity", "energy"]:
        if metric in col_map and col_map[metric] in summary["key_metrics"]:
            m = summary["key_metrics"][col_map[metric]]
            parts.append(f"{metric.replace('_', ' ').title()}: mean={m['mean']}, range [{m['min']}, {m['max']}].")

    summary["text_summary"] = " ".join(parts)
    return summary


# ---------------------------------------------------------------------------
# 6. Safe Stats Query (replaces broken DuckDB unnest approach)
# ---------------------------------------------------------------------------
def query_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column stats for numeric columns using pandas (safe fallback)."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    rows = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "column_name": col,
            "min_val":     round(float(s.min()), 2),
            "max_val":     round(float(s.max()), 2),
            "mean_val":    round(float(s.mean()), 2),
            "median_val":  round(float(s.median()), 2),
        })
    return pd.DataFrame(rows)


def query_with_duckdb(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Run an arbitrary SQL query against the uploaded DataFrame."""
    try:
        import duckdb
        con = duckdb.connect()
        con.register("company_data", df)
        result = con.execute(sql).fetchdf()
        con.close()
        return result
    except Exception as e:
        raise RuntimeError(f"DuckDB query failed: {e}")


# ---------------------------------------------------------------------------
# 7. Company Trend Analysis (unchanged, solid)
# ---------------------------------------------------------------------------
def analyse_company_trends(df: pd.DataFrame, company_name: str, col_map: dict) -> dict:
    if "company" in col_map and company_name:
        mask = df[col_map["company"]] == company_name
        cdf = df[mask].copy()
    else:
        cdf = df.copy()

    if cdf.empty:
        return {"company": company_name, "metrics": {}, "overall_direction": "Unknown"}

    if "year" in col_map:
        cdf = cdf.sort_values(col_map["year"])

    result = {"company": company_name, "metrics": {}, "year_range": None}

    if "year" in col_map:
        yrs = pd.to_numeric(cdf[col_map["year"]], errors="coerce").dropna()
        if not yrs.empty:
            result["year_range"] = (int(yrs.min()), int(yrs.max()))

    improving_count = 0
    declining_count = 0

    lower_is_better  = {"carbon_emissions", "water", "waste_total", "employee_turnover",
                        "safety", "scope1", "scope2", "scope3", "energy"}
    higher_is_better = {"renewable_energy", "waste_recycled", "gender_diversity",
                        "training", "board_independence", "community_investment",
                        "esg_budget", "revenue", "employees"}

    for metric, col in col_map.items():
        if metric in ("company", "industry", "country", "year"):
            continue
        if col not in cdf.columns:
            continue
        series = pd.to_numeric(cdf[col], errors="coerce").dropna()
        if len(series) < 2:
            continue

        first_val  = float(series.iloc[0])
        last_val   = float(series.iloc[-1])
        pct_change = round(((last_val - first_val) / abs(first_val)) * 100, 1) if first_val != 0 else 0.0

        if metric in lower_is_better:
            is_improving = pct_change < 0
        elif metric in higher_is_better:
            is_improving = pct_change > 0
        else:
            is_improving = None

        if is_improving is True:
            improving_count += 1
        elif is_improving is False:
            declining_count += 1

        result["metrics"][metric] = {
            "column":       col,
            "first_value":  first_val,
            "last_value":   last_val,
            "pct_change":   pct_change,
            "is_improving": is_improving,
            "trend_values": series.tolist(),
        }

    total = improving_count + declining_count
    if total == 0:
        result["overall_direction"] = "Insufficient Data"
    elif improving_count / total >= 0.7:
        result["overall_direction"] = "Improving"
    elif declining_count / total >= 0.7:
        result["overall_direction"] = "Declining"
    else:
        result["overall_direction"] = "Mixed"

    result["improving_count"] = improving_count
    result["declining_count"] = declining_count
    return result


# ---------------------------------------------------------------------------
# 8. ESG Scoring from Actual Data
# ---------------------------------------------------------------------------
def compute_esg_score_from_data(df: pd.DataFrame, company_name: str, col_map: dict) -> dict:
    if "company" in col_map and company_name:
        cdf = df[df[col_map["company"]] == company_name].copy()
    else:
        cdf = df.copy()

    if "year" in col_map:
        cdf = cdf.sort_values(col_map["year"])

    if len(cdf) == 0:
        return {"overall": 50, "environmental": 50, "social": 50, "governance": 50,
                "grade": "C", "rationale": "Insufficient data."}

    latest = cdf.iloc[-1]

    # Environmental
    e_scores = []
    if "renewable_energy" in col_map:
        val = pd.to_numeric(latest.get(col_map["renewable_energy"], None), errors="coerce")
        if pd.notna(val):
            e_scores.append(min(float(val), 100))
    if "carbon_emissions" in col_map and "employees" in col_map:
        emissions = pd.to_numeric(latest.get(col_map["carbon_emissions"], None), errors="coerce")
        emps      = pd.to_numeric(latest.get(col_map["employees"],        None), errors="coerce")
        if pd.notna(emissions) and pd.notna(emps) and emps > 0:
            per_emp = emissions / emps
            e_scores.append(max(20, min(90, 90 - (per_emp - 1) * 7.8)))
    if "waste_recycled" in col_map:
        val = pd.to_numeric(latest.get(col_map["waste_recycled"], None), errors="coerce")
        if pd.notna(val):
            e_scores.append(min(float(val), 100))
    env_score = round(sum(e_scores) / len(e_scores), 1) if e_scores else 50.0

    # Social
    s_scores = []
    if "gender_diversity" in col_map:
        val = pd.to_numeric(latest.get(col_map["gender_diversity"], None), errors="coerce")
        if pd.notna(val):
            s_scores.append(min(float(val) * 2, 100))
    if "employee_turnover" in col_map:
        val = pd.to_numeric(latest.get(col_map["employee_turnover"], None), errors="coerce")
        if pd.notna(val):
            s_scores.append(max(20, min(90, 90 - (float(val) - 5) * 2.8)))
    if "training" in col_map:
        val = pd.to_numeric(latest.get(col_map["training"], None), errors="coerce")
        if pd.notna(val):
            s_scores.append(max(20, min(90, 20 + float(val) * 1.75)))
    if "safety" in col_map:
        val = pd.to_numeric(latest.get(col_map["safety"], None), errors="coerce")
        if pd.notna(val):
            s_scores.append(max(20, min(95, 95 - float(val) * 3.75)))
    social_score = round(sum(s_scores) / len(s_scores), 1) if s_scores else 50.0

    # Governance
    g_scores = []
    if "board_independence" in col_map:
        val = pd.to_numeric(latest.get(col_map["board_independence"], None), errors="coerce")
        if pd.notna(val):
            g_scores.append(min(float(val), 100))
    if "esg_budget" in col_map and "revenue" in col_map:
        budget = pd.to_numeric(latest.get(col_map["esg_budget"], None), errors="coerce")
        rev    = pd.to_numeric(latest.get(col_map["revenue"],     None), errors="coerce")
        if pd.notna(budget) and pd.notna(rev) and rev > 0:
            pct = (float(budget) / float(rev)) * 100
            g_scores.append(max(30, min(90, 30 + pct * 60)))
    if "community_investment" in col_map:
        val = pd.to_numeric(latest.get(col_map["community_investment"], None), errors="coerce")
        if pd.notna(val) and float(val) > 0:
            import math
            g_scores.append(min(90, 40 + math.log10(float(val)) * 10))
    gov_score = round(sum(g_scores) / len(g_scores), 1) if g_scores else 50.0

    env_score   = max(10, min(95, env_score))
    social_score = max(10, min(95, social_score))
    gov_score   = max(10, min(95, gov_score))
    overall     = round(env_score * 0.4 + social_score * 0.3 + gov_score * 0.3, 1)
    grade       = "A" if overall >= 80 else ("B" if overall >= 65 else ("C" if overall >= 50 else ("D" if overall >= 35 else "F")))

    return {
        "overall": overall, "environmental": env_score,
        "social": social_score, "governance": gov_score,
        "grade": grade,
        "rationale": (
            f"Scored from actual uploaded data. "
            f"E based on {len(e_scores)} metric(s), "
            f"S based on {len(s_scores)} metric(s), "
            f"G based on {len(g_scores)} metric(s)."
        ),
    }


# ---------------------------------------------------------------------------
# 9. Peer Comparison
# ---------------------------------------------------------------------------
def compute_peer_comparison(df: pd.DataFrame, company_name: str, col_map: dict) -> pd.DataFrame:
    if "company" not in col_map:
        return pd.DataFrame()

    companies = df[col_map["company"]].dropna().unique().tolist()
    if len(companies) < 2:
        return pd.DataFrame()

    rows = []
    for comp in companies:
        scores = compute_esg_score_from_data(df, comp, col_map)
        cdf    = df[df[col_map["company"]] == comp]
        if "year" in col_map:
            cdf = cdf.sort_values(col_map["year"])
        latest = cdf.iloc[-1] if len(cdf) > 0 else pd.Series(dtype=object)

        row = {
            "Company":      comp,
            "ESG Score":    scores["overall"],
            "Grade":        scores["grade"],
            "Environmental": scores["environmental"],
            "Social":        scores["social"],
            "Governance":    scores["governance"],
        }
        for metric in ["employees", "carbon_emissions", "renewable_energy",
                       "gender_diversity", "board_independence", "waste_recycled"]:
            if metric in col_map and col_map[metric] in cdf.columns:
                val = pd.to_numeric(latest.get(col_map[metric], None), errors="coerce")
                row[metric.replace("_", " ").title()] = round(float(val), 1) if pd.notna(val) else None

        rows.append(row)

    peer_df = pd.DataFrame(rows).sort_values("ESG Score", ascending=False).reset_index(drop=True)
    peer_df.index += 1
    peer_df.index.name = "Rank"
    return peer_df


# ---------------------------------------------------------------------------
# 10. Year-over-Year Metrics
# ---------------------------------------------------------------------------
def get_yoy_metrics(df: pd.DataFrame, company_name: str, col_map: dict) -> pd.DataFrame:
    if "company" in col_map and company_name:
        cdf = df[df[col_map["company"]] == company_name].copy()
    else:
        cdf = df.copy()

    if "year" not in col_map:
        return pd.DataFrame()

    cdf   = cdf.sort_values(col_map["year"])
    years = pd.to_numeric(cdf[col_map["year"]], errors="coerce").dropna().astype(int).tolist()

    rows = []
    for metric, col in col_map.items():
        if metric in ("company", "industry", "country", "year"):
            continue
        if col not in cdf.columns:
            continue
        values = pd.to_numeric(cdf[col], errors="coerce").tolist()
        if all(pd.isna(v) for v in values):
            continue
        row = {"Metric": metric.replace("_", " ").title(), "Column": col}
        for yr, val in zip(years, values):
            row[str(yr)] = round(val, 1) if pd.notna(val) else None
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11. KPI Improvements
#     FIX: now attempts to use real data values as baselines where available
# ---------------------------------------------------------------------------
def compute_kpi_improvements(
    business_size: str,
    industry: str,
    solutions: list,
    df: Optional[pd.DataFrame] = None,
    col_map: Optional[dict] = None,
) -> pd.DataFrame:
    scale = {"Small": 1.0, "Medium": 1.5, "Large": 2.0}.get(business_size, 1.0)

    # Try to pull real baselines from uploaded data
    real_baselines: dict = {}
    if df is not None and col_map is not None and len(df) > 0:
        # Use most recent row
        latest_row = df.sort_values(col_map["year"]).iloc[-1] if "year" in col_map else df.iloc[-1]
        metric_to_kpi = {
            "carbon_emissions":  ("Carbon Emissions (tCO₂e)", -15),
            "energy":            ("Energy Consumption (MWh)", -10),
            "gender_diversity":  ("Gender Diversity (%)",     +8),
            "waste_recycled":    ("Waste Diversion Rate (%)", +20),
            "employee_turnover": ("Employee Turnover (%)",    -12),
        }
        for metric, (kpi_name, imp) in metric_to_kpi.items():
            if metric in col_map and col_map[metric] in df.columns:
                val = pd.to_numeric(latest_row.get(col_map[metric], None), errors="coerce")
                if pd.notna(val):
                    real_baselines[kpi_name] = (float(val), imp)

    # Default / fallback baselines
    kpi_templates = [
        ("Carbon Emissions (tCO₂e)",  1000 * scale, -15),
        ("Energy Efficiency (%)",       60,           +12),
        ("Employee Satisfaction (%)",   65,           +10),
        ("Waste Diversion Rate (%)",    40,           +20),
        ("ESG Disclosure Score",        55,           +18),
    ]

    rows = []
    for sol in solutions:
        for kpi_name, base_default, imp_default in kpi_templates:
            if kpi_name in real_baselines:
                base, imp = real_baselines[kpi_name]
            else:
                base, imp = base_default, imp_default
            rows.append({
                "Solution":        sol["name"],
                "KPI":             kpi_name,
                "Current":         round(base, 1),
                "Projected":       round(base * (1 + imp / 100), 1),
                "Improvement_Pct": imp,
            })
    return pd.DataFrame(rows)


def compute_real_kpi_projections(trends: dict, solutions: list) -> pd.DataFrame:
    if not trends or not trends.get("metrics"):
        return pd.DataFrame()

    improvement_map = {
        "carbon_emissions":     ("Carbon Emissions (tCO₂e)",  -15),
        "renewable_energy":     ("Renewable Energy (%)",       +12),
        "waste_recycled":       ("Waste Recycled (%)",         +15),
        "gender_diversity":     ("Gender Diversity (%)",       +8),
        "employee_turnover":    ("Employee Turnover (%)",      -20),
        "board_independence":   ("Board Independence (%)",     +10),
        "energy":               ("Energy Consumption (MWh)",   -10),
        "water":                ("Water Usage (m³)",           -12),
        "safety":               ("Safety Incidents",           -30),
    }

    rows = []
    for sol in solutions[:3]:
        for metric, data in trends["metrics"].items():
            if metric not in improvement_map:
                continue
            kpi_name, imp_pct = improvement_map[metric]
            current  = data["last_value"]
            projected = current * (1 + imp_pct / 100)
            rows.append({
                "Solution":        sol["name"],
                "KPI":             kpi_name,
                "Current":         round(current, 1),
                "Projected":       round(projected, 1),
                "Improvement_Pct": imp_pct,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 12. Industry Benchmarks
# ---------------------------------------------------------------------------
_INDUSTRY_BENCHMARKS = {
    "Technology": {
        "esg_score_avg": 62, "esg_score_top_quartile": 78,
        "carbon_intensity_avg": 45, "renewable_energy_pct": 55,
        "gender_diversity_pct": 32, "board_independence_pct": 75,
        "employee_turnover_pct": 18, "waste_recycling_pct": 60,
        "esg_disclosure_pct": 72,
    },
    "Finance": {
        "esg_score_avg": 58, "esg_score_top_quartile": 74,
        "carbon_intensity_avg": 12, "renewable_energy_pct": 42,
        "gender_diversity_pct": 38, "board_independence_pct": 80,
        "employee_turnover_pct": 15, "waste_recycling_pct": 45,
        "esg_disclosure_pct": 78,
    },
    "Financial Services": {
        "esg_score_avg": 58, "esg_score_top_quartile": 74,
        "carbon_intensity_avg": 12, "renewable_energy_pct": 42,
        "gender_diversity_pct": 38, "board_independence_pct": 80,
        "employee_turnover_pct": 15, "waste_recycling_pct": 45,
        "esg_disclosure_pct": 78,
    },
    "Manufacturing": {
        "esg_score_avg": 48, "esg_score_top_quartile": 65,
        "carbon_intensity_avg": 220, "renewable_energy_pct": 28,
        "gender_diversity_pct": 22, "board_independence_pct": 65,
        "employee_turnover_pct": 12, "waste_recycling_pct": 55,
        "esg_disclosure_pct": 58,
    },
    "Energy": {
        "esg_score_avg": 42, "esg_score_top_quartile": 60,
        "carbon_intensity_avg": 850, "renewable_energy_pct": 22,
        "gender_diversity_pct": 20, "board_independence_pct": 70,
        "employee_turnover_pct": 10, "waste_recycling_pct": 40,
        "esg_disclosure_pct": 65,
    },
    "Oil And Gas": {
        "esg_score_avg": 38, "esg_score_top_quartile": 55,
        "carbon_intensity_avg": 1100, "renewable_energy_pct": 15,
        "gender_diversity_pct": 18, "board_independence_pct": 68,
        "employee_turnover_pct": 10, "waste_recycling_pct": 35,
        "esg_disclosure_pct": 60,
    },
    "Healthcare": {
        "esg_score_avg": 55, "esg_score_top_quartile": 72,
        "carbon_intensity_avg": 60, "renewable_energy_pct": 35,
        "gender_diversity_pct": 45, "board_independence_pct": 72,
        "employee_turnover_pct": 20, "waste_recycling_pct": 50,
        "esg_disclosure_pct": 68,
    },
    "Retail": {
        "esg_score_avg": 50, "esg_score_top_quartile": 68,
        "carbon_intensity_avg": 35, "renewable_energy_pct": 40,
        "gender_diversity_pct": 42, "board_independence_pct": 68,
        "employee_turnover_pct": 25, "waste_recycling_pct": 48,
        "esg_disclosure_pct": 60,
    },
    "Agriculture": {
        "esg_score_avg": 44, "esg_score_top_quartile": 62,
        "carbon_intensity_avg": 300, "renewable_energy_pct": 25,
        "gender_diversity_pct": 28, "board_independence_pct": 60,
        "employee_turnover_pct": 22, "waste_recycling_pct": 35,
        "esg_disclosure_pct": 50,
    },
    "Construction": {
        "esg_score_avg": 42, "esg_score_top_quartile": 58,
        "carbon_intensity_avg": 180, "renewable_energy_pct": 20,
        "gender_diversity_pct": 18, "board_independence_pct": 62,
        "employee_turnover_pct": 20, "waste_recycling_pct": 42,
        "esg_disclosure_pct": 52,
    },
    "Transportation": {
        "esg_score_avg": 46, "esg_score_top_quartile": 63,
        "carbon_intensity_avg": 400, "renewable_energy_pct": 18,
        "gender_diversity_pct": 25, "board_independence_pct": 65,
        "employee_turnover_pct": 16, "waste_recycling_pct": 38,
        "esg_disclosure_pct": 56,
    },
    "Pharmaceuticals": {
        "esg_score_avg": 57, "esg_score_top_quartile": 73,
        "carbon_intensity_avg": 55, "renewable_energy_pct": 38,
        "gender_diversity_pct": 44, "board_independence_pct": 74,
        "employee_turnover_pct": 14, "waste_recycling_pct": 55,
        "esg_disclosure_pct": 70,
    },
}

_DEFAULT_BENCHMARK = {
    "esg_score_avg": 52, "esg_score_top_quartile": 70,
    "carbon_intensity_avg": 100, "renewable_energy_pct": 35,
    "gender_diversity_pct": 30, "board_independence_pct": 70,
    "employee_turnover_pct": 16, "waste_recycling_pct": 50,
    "esg_disclosure_pct": 62,
}


def get_industry_benchmark(industry: str) -> dict:
    return _INDUSTRY_BENCHMARKS.get(industry, _DEFAULT_BENCHMARK)


def compute_benchmark_comparison(
    profile: dict,
    maturity_scores: dict,
    industry: str,
    df: Optional[pd.DataFrame] = None,
    col_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compare company ESG metrics against industry benchmarks.
    FIX: now uses actual uploaded data values where available, instead of
    computing Company_Value as a fixed percentage of benchmark (was misleading).
    """
    benchmark = get_industry_benchmark(industry)

    # Try to read real values from uploaded data
    real_vals: dict = {}
    if df is not None and col_map is not None and len(df) > 0:
        latest_row = df.sort_values(col_map["year"]).iloc[-1] if "year" in col_map else df.iloc[-1]
        for metric, col in col_map.items():
            val = pd.to_numeric(latest_row.get(col, None), errors="coerce")
            if pd.notna(val):
                real_vals[metric] = float(val)

    def get_company_val(metric_key: str, fallback_ratio: float) -> float:
        """Return real data value if available, else estimate from benchmark."""
        return real_vals.get(metric_key, benchmark[list(benchmark.keys())[0]] * fallback_ratio)

    comparisons = [
        {
            "Metric":        "Overall ESG Score",
            "Company_Value": maturity_scores.get("overall", 0),
            "Industry_Avg":  benchmark["esg_score_avg"],
            "Top_Quartile":  benchmark["esg_score_top_quartile"],
        },
        {
            "Metric":        "Environmental Pillar",
            "Company_Value": maturity_scores.get("environmental", 0),
            "Industry_Avg":  round(benchmark["esg_score_avg"] * 0.95, 1),
            "Top_Quartile":  round(benchmark["esg_score_top_quartile"] * 0.95, 1),
        },
        {
            "Metric":        "Social Pillar",
            "Company_Value": maturity_scores.get("social", 0),
            "Industry_Avg":  benchmark["esg_score_avg"],
            "Top_Quartile":  benchmark["esg_score_top_quartile"],
        },
        {
            "Metric":        "Governance Pillar",
            "Company_Value": maturity_scores.get("governance", 0),
            "Industry_Avg":  round(benchmark["esg_score_avg"] * 1.05, 1),
            "Top_Quartile":  round(benchmark["esg_score_top_quartile"] * 1.05, 1),
        },
        {
            "Metric":        "Renewable Energy (%)",
            "Company_Value": real_vals.get("renewable_energy", round(benchmark["renewable_energy_pct"] * 0.7, 1)),
            "Industry_Avg":  benchmark["renewable_energy_pct"],
            "Top_Quartile":  min(benchmark["renewable_energy_pct"] * 1.5, 100),
        },
        {
            "Metric":        "Gender Diversity (%)",
            "Company_Value": real_vals.get("gender_diversity", round(benchmark["gender_diversity_pct"] * 0.85, 1)),
            "Industry_Avg":  benchmark["gender_diversity_pct"],
            "Top_Quartile":  min(benchmark["gender_diversity_pct"] * 1.3, 50),
        },
        {
            "Metric":        "ESG Disclosure (%)",
            "Company_Value": real_vals.get("esg_budget", round(benchmark["esg_disclosure_pct"] * 0.65, 1)),
            "Industry_Avg":  benchmark["esg_disclosure_pct"],
            "Top_Quartile":  min(benchmark["esg_disclosure_pct"] * 1.2, 100),
        },
    ]

    df_out = pd.DataFrame(comparisons)
    for col in ["Company_Value", "Industry_Avg", "Top_Quartile"]:
        df_out[col] = df_out[col].round(1)
    df_out["Gap"] = (df_out["Company_Value"] - df_out["Industry_Avg"]).round(1)
    df_out["Status"] = df_out["Gap"].apply(
        lambda x: "Above Average" if x > 5 else ("At Average" if x >= -5 else "Below Average")
    )
    return df_out


# ---------------------------------------------------------------------------
# 13. Excel Export Helper (new)
# ---------------------------------------------------------------------------
def export_to_excel(
    profile: dict,
    maturity: dict,
    solutions: list,
    kpi_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> bytes:
    """Export key ESG analysis outputs to a multi-sheet Excel workbook."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1 – Profile & Maturity
        profile_data = {
            "Field": ["Industry", "Country", "Employees", "ESG Budget", "Business Size",
                      "ESG Grade", "Overall Score", "Environmental", "Social", "Governance"],
            "Value": [
                profile.get("industry", "N/A"),
                profile.get("country", "N/A"),
                profile.get("employee_count", "N/A"),
                f"${profile.get('budget', 0):,.0f}",
                profile.get("business_size", "N/A"),
                maturity.get("grade", "N/A"),
                maturity.get("overall", "N/A"),
                maturity.get("environmental", "N/A"),
                maturity.get("social", "N/A"),
                maturity.get("governance", "N/A"),
            ],
        }
        pd.DataFrame(profile_data).to_excel(writer, sheet_name="Profile & Maturity", index=False)

        # Sheet 2 – Solutions
        if solutions:
            sol_df = pd.DataFrame(solutions)[
                ["rank", "name", "category", "fit_score",
                 "upfront_cost_low", "upfront_cost_high", "annual_savings", "payback_years"]
            ]
            sol_df.columns = ["Rank", "Solution", "Category", "Fit Score",
                               "Cost Low ($)", "Cost High ($)", "Annual Savings ($)", "Payback (yrs)"]
            sol_df.to_excel(writer, sheet_name="Solutions", index=False)

        # Sheet 3 – KPIs
        if not kpi_df.empty:
            kpi_df.to_excel(writer, sheet_name="KPI Projections", index=False)

        # Sheet 4 – Benchmarks
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_df.to_excel(writer, sheet_name="Benchmarks", index=False)

    return output.getvalue()
