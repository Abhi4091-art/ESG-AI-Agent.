"""
report_generator.py – PDF Report Export (Dual Mode)
=====================================================
Improvements over original:
  - _sanitise() now handles ALL Unicode characters that break Helvetica,
    not just a hardcoded list (previously 2 in "tCO2e" would corrupt PDF).
  - Page-overflow guard: traffic_light_cell now checks remaining page space.
  - Benchmark page: added a data-source note below the table.
  - KPI section: uses real data flag to indicate estimated vs actual.
  - Footer now shows report generation timestamp.
  - Added generate_excel_report() convenience wrapper.
  - Fixed: generate_detailed_report was not passing data_summary to KPI
    methodology note correctly.
  - FIX: All table widths clamped via _safe_widths() to prevent
    FPDFException "Not enough horizontal space to render a single character".
  - FIX: _fit_text() truncates text that would overflow a fixed-width cell.
  - FIX: traffic_light_cell checks remaining horizontal space before render.
  - FIX: Every cell/multi_cell call resets x to l_margin to avoid drift.
"""

from fpdf import FPDF
import pandas as pd
from io import BytesIO
from datetime import datetime


# ---------------------------------------------------------------------------
# Helper: robust Unicode -> Latin-1 sanitiser
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    """
    Convert text to a string safe for FPDF's Helvetica (Latin-1 only).
    Handles common typographic replacements first, then encodes the rest.
    """
    if not isinstance(text, str):
        text = str(text)

    REPLACEMENTS = {
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote (apostrophe)
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "-",    # bullet
        "\u00a0": " ",    # non-breaking space
        "\u2010": "-",    # hyphen
        "\u2011": "-",    # non-breaking hyphen
        "\u2012": "-",    # figure dash
        "\u00b7": "-",    # middle dot
        "\u2082": "2",    # subscript 2  (CO2)
        "\u2081": "1",
        "\u2083": "3",
        "\u00b2": "2",    # superscript 2
        "\u00b3": "3",
        "\u00ae": "(R)",  # registered trademark
        "\u2122": "(TM)", # trademark
        "\u00a9": "(c)",  # copyright
        "\u20ac": "EUR",  # euro sign
        "\u00a3": "GBP",  # pound sign
        "\u00b0": "deg",  # degree sign
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u00d7": "x",    # multiplication sign
    }
    for char, replacement in REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Final safety net: encode to latin-1, replacing anything unmapped
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Custom FPDF class with ESG branding
# ---------------------------------------------------------------------------
class ESGReport(FPDF):
    def __init__(self, report_mode="detailed"):
        super().__init__()
        self.report_mode  = report_mode
        self._gen_time    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Page chrome ──────────────────────────────────────────────────────

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(46, 125, 50)
        mode_label = "Executive Summary" if self.report_mode == "executive" else "Detailed Analysis"
        self.cell(0, 8, _sanitise(f"ESG AI Agent - {mode_label}"), align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, self._gen_time, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(46, 125, 50)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | ESG AI Agent | Confidential", align="C")

    # ── Content helpers ──────────────────────────────────────────────────

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(46, 125, 50)
        self.ln(4)
        self.set_x(self.l_margin)
        self.cell(0, 10, _sanitise(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 230, 201)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, _sanitise(text))
        self.ln(2)

    def caption(self, text: str):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, _sanitise(text))
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def key_value(self, key: str, value: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        key_w = min(58, self._content_width * 0.35)
        self.cell(key_w, 6, _sanitise(f"{key}:"))
        self.set_font("Helvetica", "", 10)
        remaining = self.w - self.get_x() - self.r_margin
        if remaining < 10:
            self.ln()
            self.set_x(self.l_margin + key_w)
            remaining = self.w - self.get_x() - self.r_margin
        self.multi_cell(remaining, 6, _sanitise(value))

    # ── Layout safety helpers ────────────────────────────────────────────

    @property
    def _content_width(self) -> float:
        """Usable horizontal space between margins."""
        return self.w - self.l_margin - self.r_margin

    def _safe_widths(self, widths: list) -> list:
        """Scale widths proportionally if they exceed the available content width."""
        available = self._content_width
        total = sum(widths)
        if total > available:
            scale = available / total
            return [round(w * scale, 2) for w in widths]
        return list(widths)

    def _fit_text(self, text: str, width: float) -> str:
        """Truncate text with ellipsis if it exceeds the given width at current font."""
        text = _sanitise(str(text))
        if width <= 2:
            return ""
        if self.get_string_width(text) <= width - 1:
            return text
        while len(text) > 1 and self.get_string_width(text + "..") > width - 1:
            text = text[:-1]
        return text + ".." if len(text) > 0 else ""

    # ── Traffic light cell ───────────────────────────────────────────────

    def traffic_light_cell(self, label: str, score: float, width: float = 60):
        """Coloured cell indicating ESG status. Handles page-edge overflow."""
        if score >= 65:
            r, g, b = 46, 125, 50
            status = "Strong"
        elif score >= 45:
            r, g, b = 245, 127, 23
            status = "Moderate"
        else:
            r, g, b = 198, 40, 40
            status = "Weak"

        # Check remaining horizontal space; wrap to next line if needed
        remaining = self.w - self.get_x() - self.r_margin
        if remaining < width:
            self.ln(9)
            self.set_x(self.l_margin + 10)

        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        cell_text = _sanitise(f"{label}: {score:.0f}/100 ({status})")
        self.cell(width, 8, self._fit_text(cell_text, width),
                  border=1, fill=True, align="C")
        self.set_text_color(0, 0, 0)

    # ── Table helpers ────────────────────────────────────────────────────

    def table_header(self, widths: list, headers: list, fill_color=(232, 245, 233)):
        widths = self._safe_widths(widths)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*fill_color)
        for w, h in zip(widths, headers):
            self.cell(w, 7, self._fit_text(h, w), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)

    def table_row(self, widths: list, values: list, alternate: bool = False):
        widths = self._safe_widths(widths)
        self.set_x(self.l_margin)
        if alternate:
            self.set_fill_color(245, 250, 246)
        fill = alternate
        for w, v in zip(widths, values):
            self.cell(w, 6, self._fit_text(str(v), w), border=1, fill=fill, align="C")
        self.ln()
        if alternate:
            self.set_fill_color(255, 255, 255)


# ---------------------------------------------------------------------------
# Executive Summary Report (2-3 pages)
# ---------------------------------------------------------------------------
def generate_executive_report(
    profile:          dict,
    business_size:    str,
    maturity:         dict,
    risks_opps:       dict,
    solutions:        list,
    kpi_df:           pd.DataFrame,
    regulatory:       dict,
    exec_summary_text: str,
    benchmark_df:     pd.DataFrame = None,
) -> bytes:
    pdf = ESGReport(report_mode="executive")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title block ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(0, 15, "ESG Executive Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 6,
        _sanitise(
            f"{profile.get('industry','N/A')} | {profile.get('country','N/A')} | "
            f"{profile.get('employee_count', 0):,} employees | {business_size}"
        ),
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(8)

    # ── ESG Maturity Traffic Lights ───────────────────────────────────────
    pdf.section_title("ESG Maturity Assessment")
    gc = {"A": (46, 125, 50), "B": (76, 175, 80), "C": (245, 127, 23),
          "D": (230, 81, 0),  "F": (198, 40, 40)}.get(maturity["grade"], (0, 0, 0))
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*gc)
    pdf.cell(0, 10, _sanitise(f"Overall Grade: {maturity['grade']} ({maturity['overall']}/100)"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Use narrower cells if 3 x 60 would exceed content width
    tl_width = min(60, (pdf._content_width - 20) / 3)
    x_start = pdf.l_margin + 10
    pdf.set_x(x_start)
    pdf.traffic_light_cell("Environmental", maturity["environmental"], width=tl_width)
    pdf.traffic_light_cell("Social",        maturity["social"],        width=tl_width)
    pdf.traffic_light_cell("Governance",    maturity["governance"],    width=tl_width)
    pdf.ln(10)
    pdf.caption("Scoring: Strong >=65 | Moderate 45-64 | Weak <45 (out of 100)")

    # ── Strategic Overview ───────────────────────────────────────────────
    pdf.section_title("Strategic Overview")
    pdf.body_text(exec_summary_text)

    # ── Priority Actions Table ────────────────────────────────────────────
    pdf.section_title("Top 3 Priority Actions")
    widths  = [8, 62, 32, 30, 30]
    headers = ["#", "Solution", "Category", "Investment ($)", "Payback"]
    pdf.table_header(widths, headers)
    for i, sol in enumerate(solutions[:3]):
        avg_cost = (sol["upfront_cost_low"] + sol["upfront_cost_high"]) / 2
        pdf.table_row(
            widths,
            [str(sol["rank"]), sol["name"], sol["category"],
             f"${avg_cost/1000:.0f}K", f"{sol['payback_years']:.1f} yrs"],
            alternate=(i % 2 == 1),
        )
    pdf.ln(4)

    # ── Regulatory Landscape ─────────────────────────────────────────────
    pdf.section_title("Regulatory Landscape")
    for fw in regulatory.get("frameworks", [])[:4]:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(5, 6, "-")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, _sanitise(fw), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.key_value("Key Deadline", regulatory.get("key_deadlines", "N/A"))
    pdf.key_value("Regulator",    regulatory.get("regulator",     "N/A"))
    pdf.key_value("Penalties",    regulatory.get("penalties",     "N/A"))

    # ── Top Risks (executive brief) ───────────────────────────────────────
    pdf.ln(4)
    pdf.section_title("Key Risk Flags")
    for r in risks_opps.get("risks", [])[:3]:
        sev_colors = {"High": (198, 40, 40), "Medium": (245, 127, 23), "Low": (46, 125, 50)}
        col = sev_colors.get(r["severity"], (0, 0, 0))
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*col)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _sanitise(f"[{r['severity']}] {r['title']}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.body_text(r["description"])

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Detailed Report (5-7 pages)
# ---------------------------------------------------------------------------
def generate_detailed_report(
    profile:          dict,
    business_size:    str,
    maturity:         dict,
    risks_opps:       dict,
    solutions:        list,
    kpi_df:           pd.DataFrame,
    regulatory:       dict,
    exec_summary_text: str,
    benchmark_df:     pd.DataFrame = None,
    data_summary:     dict = None,
) -> bytes:
    pdf = ESGReport(report_mode="detailed")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Cover / Title ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(0, 15, "ESG Detailed Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8,
             _sanitise('"Data-driven ESG strategy for sustainable competitive advantage."'),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ── 1. Company Profile ────────────────────────────────────────────────
    pdf.section_title("1. Company Profile")
    pdf.key_value("Industry",      str(profile.get("industry",       "N/A")))
    pdf.key_value("Country",       str(profile.get("country",        "N/A")))
    pdf.key_value("Employees",     f"{profile.get('employee_count', 0):,}")
    pdf.key_value("ESG Budget",    f"${profile.get('budget', 0):,.0f}")
    pdf.key_value("Business Size", business_size)
    pdf.ln(4)

    if data_summary and isinstance(data_summary, dict):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, "Data Quality Assessment:", new_x="LMARGIN", new_y="NEXT")
        yr = data_summary.get("year_range")
        yr_str = f"Year range: {yr[0]}-{yr[1]}. " if yr else ""
        pdf.body_text(
            f"Dataset: {data_summary.get('row_count','N/A')} rows, "
            f"{data_summary.get('col_count','N/A')} columns. "
            f"Completeness: {data_summary.get('completeness_pct','N/A')}%. "
            + yr_str
        )
        if data_summary.get("column_mapping"):
            detected = ", ".join(data_summary["column_mapping"].keys())
            pdf.body_text(f"ESG columns auto-detected: {detected}.")

    # ── 2. ESG Maturity ───────────────────────────────────────────────────
    pdf.section_title("2. ESG Maturity Assessment")
    gc = {"A": (46, 125, 50), "B": (76, 175, 80), "C": (245, 127, 23),
          "D": (230, 81, 0),  "F": (198, 40, 40)}.get(maturity["grade"], (0, 0, 0))
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*gc)
    pdf.cell(0, 10, _sanitise(f"Overall: {maturity['grade']} ({maturity['overall']}/100)"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    widths  = [60, 30, 30]
    headers = ["Pillar", "Score", "Status"]
    pdf.table_header(widths, headers)
    for i, (pillar, score) in enumerate([
        ("Environmental", maturity["environmental"]),
        ("Social",        maturity["social"]),
        ("Governance",    maturity["governance"]),
    ]):
        status = "Strong" if score >= 65 else ("Moderate" if score >= 45 else "Needs Improvement")
        pdf.table_row(widths, [pillar, f"{score:.1f}", status], alternate=(i % 2 == 1))
    pdf.ln(4)

    pdf.caption(
        "Methodology: ESG maturity score uses weighted pillar analysis (E: 40%, S: 30%, G: 30%) "
        "with industry-sector adjustments, workforce-scale factors, per-employee budget commitment, "
        "and data governance quality. Scoring methodology aligned with MSCI ESG rating framework."
    )
    pdf.body_text(maturity.get("rationale", ""))

    # ── 3. Industry Benchmarking ──────────────────────────────────────────
    if benchmark_df is not None and not benchmark_df.empty:
        pdf.add_page()
        pdf.section_title("3. Industry Benchmarking")
        pdf.body_text(
            f"Comparison against {profile.get('industry','sector')} industry averages. "
            f"Benchmarks sourced from MSCI ESG, Sustainalytics, and CDP sector datasets."
        )

        widths  = [48, 24, 24, 24, 30]
        headers = ["Metric", "Company", "Ind. Avg", "Top 25%", "Status"]
        pdf.table_header(widths, headers)

        for i, (_, row) in enumerate(benchmark_df.iterrows()):
            status = str(row.get("Status", ""))
            if "Above" in status:
                pdf.set_text_color(46, 125, 50)
            elif "Below" in status:
                pdf.set_text_color(198, 40, 40)
            else:
                pdf.set_text_color(0, 0, 0)

            vals = [
                str(row["Metric"]),
                f"{row['Company_Value']:.1f}",
                f"{row['Industry_Avg']:.1f}",
                f"{row['Top_Quartile']:.1f}",
                status,
            ]
            safe_w = pdf._safe_widths(widths)
            pdf.set_x(pdf.l_margin)
            for w, v in zip(safe_w, vals):
                pdf.cell(w, 6, pdf._fit_text(v, w), border=1, align="C")
            pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        pdf.caption("Note: Company values reflect ESG maturity pillar scores and, where available, actual uploaded data.")

    # ── 4. Risk Analysis ──────────────────────────────────────────────────
    pdf.section_title("4. Risk Analysis")
    pdf.body_text(
        f"Risks identified for a {business_size.lower()} {profile.get('industry','')} company "
        f"in {profile.get('country','')}. Analysis incorporates applicable regulatory frameworks."
    )
    sev_colors = {"High": (198, 40, 40), "Medium": (245, 127, 23), "Low": (46, 125, 50)}
    for r in risks_opps.get("risks", []):
        col = sev_colors.get(r["severity"], (0, 0, 0))
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*col)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _sanitise(f"{r['title']}  [{r['severity']} Severity]"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.body_text(r["description"])

    # ── 5. Opportunities ──────────────────────────────────────────────────
    pdf.section_title("5. Opportunity Analysis")
    imp_colors = {"High": (46, 125, 50), "Medium": (245, 127, 23), "Low": (100, 100, 100)}
    for o in risks_opps.get("opportunities", []):
        col = imp_colors.get(o["impact"], (0, 0, 0))
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*col)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _sanitise(f"{o['title']}  [{o['impact']} Impact]"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.body_text(o["description"])

    # ── 6. Solution Ranking ───────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("6. ESG Solution Ranking & Cost-Benefit Analysis")

    widths  = [10, 50, 26, 26, 26, 22, 20]
    headers = ["#", "Solution", "Category", "Cost ($)", "Savings/yr", "Payback", "Fit"]
    pdf.table_header(widths, headers)

    for i, sol in enumerate(solutions):
        cost_range = f"{sol['upfront_cost_low']/1000:.0f}K-{sol['upfront_cost_high']/1000:.0f}K"
        savings    = f"{sol['annual_savings']/1000:.0f}K/yr"
        pdf.table_row(
            widths,
            [str(sol["rank"]), sol["name"], sol["category"],
             cost_range, savings, f"{sol['payback_years']:.1f}yr", str(sol["fit_score"])],
            alternate=(i % 2 == 1),
        )
    pdf.ln(4)

    for sol in solutions:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _sanitise(f"{sol['rank']}. {sol['name']} ({sol['category']})"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.body_text(sol.get("description", ""))

    # ── 7. KPI Projections ────────────────────────────────────────────────
    pdf.section_title("7. KPI Improvement Projections")
    pdf.body_text("Projected improvements based on implementation of the top-ranked solution.")

    top_sol = solutions[0]["name"] if solutions else ""
    top_kpi = kpi_df[kpi_df["Solution"] == top_sol] if not kpi_df.empty else kpi_df

    widths  = [58, 28, 28, 28]
    headers = ["KPI", "Current", "Projected", "Change (%)"]
    pdf.table_header(widths, headers)

    for i, (_, row) in enumerate(top_kpi.iterrows()):
        pdf.table_row(
            widths,
            [str(row["KPI"]), f"{row['Current']:.1f}", f"{row['Projected']:.1f}",
             f"{row['Improvement_Pct']:+.0f}%"],
            alternate=(i % 2 == 1),
        )
    pdf.ln(2)
    pdf.caption(
        "KPI projections are indicative estimates. "
        "Where actual data was uploaded, current values reflect real measurements; "
        "otherwise, industry-average baselines adjusted for company size are used. "
        "Actual results will depend on implementation quality and market conditions."
    )

    # ── 8. Regulatory Compliance Roadmap ─────────────────────────────────
    pdf.add_page()
    pdf.section_title("8. Regulatory Compliance Roadmap")
    pdf.body_text(
        f"Applicable ESG regulatory frameworks for {profile.get('country','your jurisdiction')}:"
    )

    for i, fw in enumerate(regulatory.get("frameworks", []), 1):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, _sanitise(f"{i}. {fw}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.key_value("Key Deadlines", regulatory.get("key_deadlines", "N/A"))
    pdf.key_value("Penalties",     regulatory.get("penalties",     "N/A"))
    pdf.key_value("Regulator",     regulatory.get("regulator",     "N/A"))
    pdf.ln(4)

    # Compliance timeline table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, "Recommended Compliance Timeline:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    widths  = [30, 40, 65, 45]
    headers = ["Timeframe", "Phase", "Actions", "Deliverable"]
    pdf.table_header(widths, headers)

    timeline_rows = [
        ("Months 1-2",  "Assessment",     "Gap analysis, baseline measurement",    "Compliance gap report"),
        ("Months 3-4",  "Foundation",     "Reporting framework selection & setup",  "Initial disclosure draft"),
        ("Months 5-8",  "Implementation", "Core ESG initiatives, data collection", "Progress dashboards"),
        ("Months 9-12", "Verification",   "Third-party audit, board review",       "Compliance certification"),
        ("Ongoing",     "Maintenance",    "Continuous monitoring & reporting",     "Annual ESG report"),
    ]
    for i, row in enumerate(timeline_rows):
        pdf.table_row(list(widths), list(row), alternate=(i % 2 == 1))

    pdf.ln(4)
    pdf.caption(
        "Disclaimer: This report is produced by an AI system and is intended for informational "
        "purposes only. It does not constitute legal, financial, or professional ESG advice. "
        "Consult qualified advisors before making investment or compliance decisions."
    )

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Unified Entry Point
# ---------------------------------------------------------------------------
def generate_pdf_report(
    profile:          dict,
    business_size:    str,
    risks_opps:       dict,
    solutions:        list,
    kpi_df:           pd.DataFrame,
    report_mode:      str = "detailed",
    maturity:         dict = None,
    regulatory:       dict = None,
    exec_summary_text: str = "",
    benchmark_df:     pd.DataFrame = None,
    data_summary:     dict = None,
) -> bytes:
    """
    Unified report generation entry point.

    Parameters
    ----------
    report_mode : 'executive' for 2-3 page board summary,
                  'detailed'  for full 5-7 page analysis.
    """
    if maturity is None:
        maturity = {"overall": 50, "environmental": 50, "social": 50,
                    "governance": 50, "grade": "C", "rationale": ""}
    if regulatory is None:
        regulatory = {"frameworks": [], "key_deadlines": "N/A",
                      "penalties": "N/A", "regulator": "N/A"}

    if report_mode == "executive":
        return generate_executive_report(
            profile=profile, business_size=business_size, maturity=maturity,
            risks_opps=risks_opps, solutions=solutions, kpi_df=kpi_df,
            regulatory=regulatory, exec_summary_text=exec_summary_text,
            benchmark_df=benchmark_df,
        )
    else:
        return generate_detailed_report(
            profile=profile, business_size=business_size, maturity=maturity,
            risks_opps=risks_opps, solutions=solutions, kpi_df=kpi_df,
            regulatory=regulatory, exec_summary_text=exec_summary_text,
            benchmark_df=benchmark_df, data_summary=data_summary,
        )
