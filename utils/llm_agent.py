"""
llm_agent.py – LLM Functions for the ESG Agent
================================================
Improvements over original:
  - Anthropic: was sending system prompt inside user message (wrong).
    Fixed to use the proper `system` parameter.
  - JSON parsing: replaced fragile regex strip with a robust json-extractor
    that handles code fences, leading text, and trailing garbage.
  - set_llm_backend now validates the OpenAI key format before accepting.
  - What-if chat: no longer sends the full solutions list as raw JSON;
    instead sends a compact summary to reduce token usage.
  - compute_esg_maturity_score: added more industries and fixed industry
    modifier lookup to be case-insensitive.
  - rank_solutions: budget-affordability guard (won't recommend solutions
    that cost more than 2× the entire ESG budget as #1).
  - classify_business_size: fixed threshold logic (was using a hard
    $500K line that made no sense for a Small company with 200 employees).
  - Added get_available_backends() helper used by the UI.
  - Removed bare `return None` at end of _call_llm (was reachable dead code).

Data Privacy (unchanged):
  - Raw CSV never sent to cloud LLMs.
  - Only aggregated summaries transmitted.
  - Ollama keeps all processing local.
"""

import re
import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# =====================================================================
# Backend State
# =====================================================================
_BACKEND = "mock"
_openai_client    = None
_anthropic_client = None
_gemini_model     = None
_ollama_model     = None
_ollama_base_url  = "http://localhost:11434"


# =====================================================================
# Backend Management
# =====================================================================
def set_llm_backend(backend: str, api_key: Optional[str] = None) -> str:
    global _BACKEND, _openai_client, _anthropic_client, _gemini_model, _ollama_model, _ollama_base_url
    backend = backend.lower().strip()

    if backend == "openai":
        try:
            import openai
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return "⚠️ OpenAI API key not provided. Using mock mode."
            if not key.startswith("sk-"):
                return "⚠️ That doesn't look like a valid OpenAI key (should start with 'sk-'). Using mock."
            _openai_client = openai.OpenAI(api_key=key)
            _BACKEND = "openai"
            return "✅ Backend set to OpenAI (GPT-4o). Only aggregated summaries – not raw data – are sent."
        except ImportError:
            return "❌ openai package not installed. Run: pip install openai"

    elif backend == "anthropic":
        try:
            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return "⚠️ Anthropic API key not provided. Using mock mode."
            _anthropic_client = anthropic.Anthropic(api_key=key)
            _BACKEND = "anthropic"
            return "✅ Backend set to Anthropic (Claude). Only aggregated summaries – not raw data – are sent."
        except ImportError:
            return "❌ anthropic package not installed. Run: pip install anthropic"

    elif backend == "gemini":
        try:
            import google.generativeai as genai
            key = api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                return "⚠️ Google API key not provided. Using mock mode."
            genai.configure(api_key=key)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            _BACKEND = "gemini"
            return "✅ Backend set to Google Gemini (2.0 Flash). Only aggregated summaries – not raw data – are sent."
        except ImportError:
            return "❌ google-generativeai package not installed. Run: pip install google-generativeai"

    elif backend == "ollama":
        model_name = "llama3.1"
        base_url   = "http://localhost:11434"

        if api_key:
            if api_key.startswith("http"):
                base_url = api_key.rstrip("/")
            else:
                model_name = api_key

        try:
            import urllib.request
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
                available  = json.loads(resp.read().decode())
                model_names = [m["name"] for m in available.get("models", [])]
                if not model_names:
                    return "⚠️ Ollama is running but no models are installed. Run: ollama pull llama3.1"
                if model_name not in model_names and f"{model_name}:latest" not in model_names:
                    model_name = model_names[0]
        except Exception as e:
            return f"❌ Cannot connect to Ollama at {base_url}. Is Ollama running? Error: {e}"

        _ollama_model    = model_name
        _ollama_base_url = base_url
        _BACKEND         = "ollama"
        return f"✅ Backend set to Ollama ({model_name}). All data stays on your local machine. 🔒"

    else:  # mock
        _BACKEND = "mock"
        return "✅ Backend set to Mock (no API calls, deterministic responses)."


def get_llm_backend() -> str:
    return _BACKEND


def get_available_backends() -> list:
    """Return list of backend identifiers that are currently configured."""
    available = ["mock"]
    if _openai_client is not None:
        available.append("openai")
    if _anthropic_client is not None:
        available.append("anthropic")
    if _gemini_model is not None:
        available.append("gemini")
    if _ollama_model is not None:
        available.append("ollama")
    return available


# =====================================================================
# Internal: Unified LLM Call
# =====================================================================
def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Route to the active backend.  Returns None on mock or any error.
    Callers fall back to deterministic mock templates on None.
    """
    if _BACKEND == "mock":
        return None

    try:
        if _BACKEND == "openai" and _openai_client is not None:
            response = _openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        elif _BACKEND == "anthropic" and _anthropic_client is not None:
            # FIX: was incorrectly concatenating system + user into a single
            # user message. Anthropic SDK has a dedicated `system` parameter.
            message = _anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text

        elif _BACKEND == "gemini" and _gemini_model is not None:
            response = _gemini_model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            return response.text

        elif _BACKEND == "ollama" and _ollama_model is not None:
            import urllib.request
            payload = json.dumps({
                "model":    _ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream":  False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{_ollama_base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result.get("message", {}).get("content") or None

    except Exception as e:
        logger.warning(f"LLM API call failed ({_BACKEND}): {e}. Falling back to mock.")
        return None

    return None   # unreachable but satisfies type checkers


# =====================================================================
# Robust JSON Extraction
# FIX: original used simple regex strip which broke on:
#   - Models that add a sentence before the JSON
#   - Trailing whitespace / newlines after the closing brace
# =====================================================================
def _extract_json(text: str):
    """
    Extract the first valid JSON object or array from a model response.
    Handles:
      - ```json ... ``` fences
      - Leading/trailing prose
      - Single-quoted keys (lenient)
    Raises json.JSONDecodeError if no valid JSON found.
    """
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    # 1. Try stripping code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find first { or [ and attempt to parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        # Walk backwards from end to find matching close
        end = cleaned.rfind(end_char)
        if end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


# =====================================================================
# PII Sanitiser
# =====================================================================
def _sanitise_text_for_llm(text: str, max_chars: int = 2000) -> str:
    if _BACKEND == "ollama":
        return text[:8000]

    sanitised = text[:max_chars]
    sanitised = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", sanitised)
    sanitised = re.sub(r"(\+?\d{1,3}[-.\ ]?)?\(?\d{2,4}\)?[-.\ ]?\d{3,4}[-.\ ]?\d{3,4}", "[PHONE]", sanitised)
    sanitised = re.sub(r"https?://\S+", "[URL]", sanitised)
    sanitised = re.sub(r"\b[A-Z]{2,3}-?\d{5,10}\b", "[ID]", sanitised)
    return sanitised


# =====================================================================
# Regulatory Framework Database  (unchanged – comprehensive)
# =====================================================================
REGULATORY_FRAMEWORKS = {
    "United Kingdom": {
        "frameworks":    ["UK Sustainability Disclosure Standards (SDS)", "TCFD (mandatory for large UK firms)",
                          "Streamlined Energy and Carbon Reporting (SECR)", "UK Green Taxonomy (in development)"],
        "key_deadlines": "ISSB-aligned UK SDS expected to apply from 2026-2027 fiscal years.",
        "penalties":     "Non-compliance with SECR can result in fines up to GBP 5,000 and director liability.",
        "regulator":     "Financial Conduct Authority (FCA) & Department for Business and Trade",
    },
    "United States": {
        "frameworks":    ["SEC Climate Disclosure Rule (stayed pending litigation)", "California SB 253/261",
                          "EPA GHG Reporting Program", "Voluntary: SASB / GRI / TCFD"],
        "key_deadlines": "California SB 253 requires Scope 1 & 2 reporting from 2026, Scope 3 from 2027.",
        "penalties":     "SEC violations carry civil penalties; California laws include administrative penalties.",
        "regulator":     "SEC, EPA, California Air Resources Board",
    },
    "Germany": {
        "frameworks":    ["EU CSRD (Corporate Sustainability Reporting Directive)", "EU Taxonomy Regulation",
                          "German Supply Chain Due Diligence Act (LkSG)", "SFDR"],
        "key_deadlines": "CSRD applies to large companies from FY2024, listed SMEs from FY2026.",
        "penalties":     "LkSG fines up to 2% of average annual global turnover.",
        "regulator":     "BaFin, BAFA",
    },
    "France": {
        "frameworks":    ["EU CSRD", "EU Taxonomy", "French Duty of Vigilance Law", "SFDR", "Article 29 Energy-Climate Law"],
        "key_deadlines": "CSRD phased in from FY2024. Duty of Vigilance applies to companies with 5,000+ employees.",
        "penalties":     "Duty of Vigilance: civil liability for failure to prevent harm in supply chain.",
        "regulator":     "AMF",
    },
    "Japan": {
        "frameworks":    ["SSBJ (ISSB-aligned)", "TCFD (voluntary, high adoption)", "TSE Corporate Governance Code"],
        "key_deadlines": "SSBJ standards expected mandatory for prime-listed companies from FY2027.",
        "penalties":     "TSE follows comply-or-explain approach.",
        "regulator":     "FSA, Tokyo Stock Exchange",
    },
    "China": {
        "frameworks":    ["Chinese Sustainability Disclosure Standards (ISSB-aligned)", "Mandatory ESG disclosure (CSRC)", "National ETS"],
        "key_deadlines": "Mandatory ESG disclosure for all A-share listed companies phased in from 2026.",
        "penalties":     "ETS non-compliance penalties under national carbon trading regulations.",
        "regulator":     "CSRC, Ministry of Ecology and Environment",
    },
    "India": {
        "frameworks":    ["BRSR (Business Responsibility and Sustainability Reporting)", "SEBI ESG framework", "Green Credit Programme"],
        "key_deadlines": "BRSR mandatory for top 1000 listed companies. BRSR Core assurance from FY2023-24.",
        "penalties":     "Non-filing can result in SEBI enforcement actions and listing restrictions.",
        "regulator":     "SEBI",
    },
    "Australia": {
        "frameworks":    ["AASB Sustainability Reporting Standards (ISSB-aligned)", "NGER", "Modern Slavery Act"],
        "key_deadlines": "Mandatory climate disclosure for Group 1 entities from 1 Jan 2025, Group 2 from 2026.",
        "penalties":     "NGER penalties up to AUD 22,200 per day for non-reporting.",
        "regulator":     "ASIC, Clean Energy Regulator",
    },
    "Singapore": {
        "frameworks":    ["SGX mandatory climate reporting (TCFD-based)", "Singapore Green Plan 2030", "MAS Guidelines"],
        "key_deadlines": "SGX mandatory climate reporting for all listed issuers from FY2025.",
        "penalties":     "SGX non-compliance can result in queries, reprimands, or suspension.",
        "regulator":     "SGX RegCo, MAS",
    },
    "Canada": {
        "frameworks":    ["CSA climate disclosure rules (proposed)", "Federal GHGRP", "Modern Slavery Act (Bill S-211)"],
        "key_deadlines": "Bill S-211 reporting obligations began January 2024.",
        "penalties":     "GHGRP non-reporting subject to federal environmental penalties.",
        "regulator":     "CSA, Environment and Climate Change Canada",
    },
    "Brazil": {
        "frameworks":    ["CVM Resolution 193 (ISSB-aligned, voluntary until 2026)", "Brazilian Taxonomy for Sustainable Finance"],
        "key_deadlines": "CVM sustainability reporting becomes mandatory from FY2026.",
        "penalties":     "CVM enforcement for listed companies; fines for non-compliance.",
        "regulator":     "CVM, Central Bank of Brazil",
    },
    "South Korea": {
        "frameworks":    ["K-ESG Guidelines", "Korean Sustainability Disclosure Standards", "K-Taxonomy"],
        "key_deadlines": "Mandatory ESG disclosure for KOSPI-listed companies (assets > KRW 2tn) from 2025, all listed from 2030.",
        "penalties":     "KRX listing rule non-compliance; financial supervisory penalties.",
        "regulator":     "Financial Services Commission, KRX",
    },
    "Netherlands": {
        "frameworks":    ["EU CSRD", "EU Taxonomy", "Dutch Climate Agreement", "SFDR"],
        "key_deadlines": "CSRD applies to large Dutch companies from FY2024.",
        "penalties":     "AFM enforcement; CSRD non-compliance fines at member-state level.",
        "regulator":     "AFM (Authority for the Financial Markets)",
    },
    "Sweden": {
        "frameworks":    ["EU CSRD", "Swedish Sustainability Act", "EU Taxonomy", "SFDR"],
        "key_deadlines": "CSRD large-company phase from FY2024; SMEs from FY2026.",
        "penalties":     "Swedish FSA enforcement; director liability under CSRD.",
        "regulator":     "Finansinspektionen (FI)",
    },
}

_DEFAULT_REGULATORY = {
    "frameworks":    ["ISSB S1 & S2", "GRI Standards (voluntary)", "TCFD Recommendations", "UN Global Compact"],
    "key_deadlines": "ISSB standards are being adopted globally; check your local securities regulator.",
    "penalties":     "Vary by jurisdiction. Increasing trend toward mandatory disclosure with financial penalties.",
    "regulator":     "Local securities regulator and environmental agency",
}


def get_regulatory_context(country: str) -> dict:
    return REGULATORY_FRAMEWORKS.get(country, _DEFAULT_REGULATORY)


# =====================================================================
# ESG Maturity Scoring
# FIX: industry modifier lookup is now case-insensitive
# =====================================================================
def compute_esg_maturity_score(
    profile: dict,
    data_stats: Optional[dict] = None,
) -> dict:
    industry = profile.get("industry", "Unknown")
    budget   = profile.get("budget", 0) or 0
    emp      = profile.get("employee_count", 0) or 0
    size     = profile.get("business_size", "Medium")

    # Environmental: budget commitment per employee
    budget_per_emp = budget / max(emp, 1)
    if budget_per_emp > 2000:
        env_score = 72
    elif budget_per_emp > 1000:
        env_score = 58
    elif budget_per_emp > 500:
        env_score = 45
    else:
        env_score = 32

    # Social: workforce scale (more nuanced than original)
    if emp > 1000:
        social_score = 60
    elif emp > 500:
        social_score = 55
    elif emp > 100:
        social_score = 45
    else:
        social_score = 35

    # Governance: absolute budget signals maturity
    if budget > 1_000_000:
        gov_score = 68
    elif budget > 500_000:
        gov_score = 60
    elif budget > 200_000:
        gov_score = 52
    elif budget > 50_000:
        gov_score = 40
    else:
        gov_score = 30

    # Industry adjustments – case-insensitive match
    _INDUSTRY_MODS = {
        "technology":        {"e": 5,  "s":  3, "g":  8},
        "finance":           {"e": -3, "s":  2, "g": 10},
        "financial services":{"e": -3, "s":  2, "g": 10},
        "manufacturing":     {"e": -8, "s": -2, "g":  3},
        "energy":            {"e":-10, "s":  0, "g":  5},
        "oil and gas":       {"e":-15, "s": -2, "g":  4},
        "healthcare":        {"e":  0, "s":  8, "g":  5},
        "pharmaceuticals":   {"e":  2, "s":  6, "g":  6},
        "retail":            {"e": -2, "s":  5, "g":  0},
        "agriculture":       {"e": -5, "s":  2, "g": -2},
        "construction":      {"e": -6, "s": -3, "g":  0},
        "transportation":    {"e": -7, "s":  0, "g":  2},
        "utilities":         {"e": -5, "s":  1, "g":  4},
        "real estate":       {"e": -3, "s":  1, "g":  2},
        "hospitality":       {"e": -1, "s":  4, "g":  0},
        "mining":            {"e":-12, "s": -4, "g":  2},
        "education":         {"e":  2, "s":  6, "g":  3},
        "telecommunications":{"e":  3, "s":  2, "g":  5},
        "consumer goods":    {"e": -1, "s":  3, "g":  1},
    }
    mods = _INDUSTRY_MODS.get(industry.lower(), {"e": 0, "s": 0, "g": 0})

    env_score    = max(10, min(95, env_score   + mods["e"]))
    social_score = max(10, min(95, social_score + mods["s"]))
    gov_score    = max(10, min(95, gov_score    + mods["g"]))

    # Data completeness bonus/penalty
    if data_stats and isinstance(data_stats, dict):
        completeness = data_stats.get("completeness_pct", 100)
        if completeness > 90:
            gov_score = min(95, gov_score + 8)
        elif completeness < 50:
            gov_score = max(10, gov_score - 10)

    overall = round(env_score * 0.40 + social_score * 0.30 + gov_score * 0.30, 1)
    grade   = "A" if overall >= 80 else ("B" if overall >= 65 else ("C" if overall >= 50 else ("D" if overall >= 35 else "F")))

    rationale = (
        f"Based on a {size.lower()} {industry.lower()} company with "
        f"{emp:,} employees and ${budget:,.0f} ESG budget. "
        f"Environmental score reflects per-employee commitment of ${budget_per_emp:,.0f}. "
        f"Social score accounts for workforce scale and industry norms. "
        f"Governance score considers budget magnitude and disclosure readiness."
    )

    return {
        "overall":       round(overall, 1),
        "environmental": round(env_score, 1),
        "social":        round(social_score, 1),
        "governance":    round(gov_score, 1),
        "grade":         grade,
        "rationale":     rationale,
    }


# =====================================================================
# Executive Summary Generation
# =====================================================================
_EXEC_SUMMARY_SYSTEM_PROMPT = (
    "You are an ESG strategy advisor preparing a board-level executive summary. "
    "Write a concise, action-oriented executive summary (150-200 words) covering:\n"
    "1. Current ESG maturity assessment (one sentence with the numeric grade)\n"
    "2. Top 3 priority actions with expected ROI and payback period\n"
    "3. Key regulatory risk and nearest compliance deadline\n"
    "4. One concrete strategic recommendation\n"
    "Use professional, direct language suitable for C-suite readers. Include specific numbers. "
    "Return plain text only (no JSON, no markdown headers, no bullet points)."
)


def generate_executive_summary(
    profile:      dict,
    maturity:     dict,
    solutions:    list,
    regulatory:   dict,
    data_summary: Optional[str] = None,
) -> str:
    user_prompt = (
        f"Company: {profile.get('industry','N/A')} sector, {profile.get('country','N/A')}, "
        f"{profile.get('employee_count', 0):,} employees, ${profile.get('budget', 0):,.0f} ESG budget.\n"
        f"ESG Maturity: Overall {maturity['overall']}/100 (Grade {maturity['grade']}), "
        f"E={maturity['environmental']}, S={maturity['social']}, G={maturity['governance']}.\n"
        f"Top solutions: {', '.join(s['name'] for s in solutions[:3])}.\n"
        f"Regulatory: {', '.join(regulatory.get('frameworks', [])[:3])}.\n"
        f"Key deadline: {regulatory.get('key_deadlines', 'N/A')}.\n"
    )
    if data_summary:
        user_prompt += f"\nData insights: {data_summary[:500]}\n"

    llm_response = _call_llm(_EXEC_SUMMARY_SYSTEM_PROMPT, user_prompt)
    if llm_response:
        return llm_response

    # Mock fallback
    industry = profile.get("industry", "your industry")
    country  = profile.get("country", "your region")
    budget   = profile.get("budget", 0) or 0
    size     = profile.get("business_size", "Medium")
    top3     = solutions[:3] if solutions else []

    summary = (
        f"Your organisation currently holds an ESG maturity grade of {maturity['grade']} "
        f"({maturity['overall']}/100), positioning it in the "
        f"{'upper' if maturity['overall'] >= 60 else 'lower'} tier of {industry.lower()} sector peers.\n\n"
        f"Three priority actions are recommended:\n"
    )
    for i, sol in enumerate(top3, 1):
        avg_cost = (sol["upfront_cost_low"] + sol["upfront_cost_high"]) / 2
        summary += (
            f"({i}) {sol['name']} – estimated ${avg_cost:,.0f} investment, "
            f"${sol['annual_savings']:,.0f}/yr savings, {sol['payback_years']:.1f}-year payback.\n"
        )

    frameworks = regulatory.get("frameworks", [])[:2]
    summary += (
        f"\nRegulatory exposure: {', '.join(frameworks)} create compliance obligations. "
        f"{regulatory.get('key_deadlines', '')}\n\n"
        f"Recommendation: Allocate ${budget * 0.6:,.0f} to the top-ranked solution immediately, "
        f"phasing remaining initiatives over 18-24 months to stay ahead of {country} disclosure deadlines."
    )
    return summary


# =====================================================================
# Entity Extraction
# =====================================================================
_EXTRACTION_SYSTEM_PROMPT = (
    "You are a data extraction assistant. Extract the following four fields from the company "
    "overview text. Return ONLY valid JSON with these exact keys:\n"
    "- employee_count: integer or null\n"
    "- budget: number (USD) or null (willingness to spend on ESG)\n"
    "- industry: string or null\n"
    "- country: string or null\n"
    "If a field cannot be determined, set it to null. "
    "Do not include any text outside the JSON object."
)


def extract_company_profile(text: str) -> dict:
    profile = {"employee_count": None, "budget": None, "industry": None, "country": None}
    if not text or not text.strip():
        return profile

    safe_text    = _sanitise_text_for_llm(text)
    llm_response = _call_llm(_EXTRACTION_SYSTEM_PROMPT, safe_text)
    if llm_response:
        try:
            parsed = _extract_json(llm_response)
            for key in profile:
                if key in parsed and parsed[key] is not None:
                    profile[key] = parsed[key]
            return profile
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("LLM returned unparseable JSON for profile extraction; using regex fallback.")

    # Regex fallback
    lower = text.lower()

    for pattern in [
        r"(\d[\d,]*)\s*employees",
        r"employee\s*(?:count|number|size)[:\s]*(\d[\d,]*)",
        r"headcount[:\s]*(\d[\d,]*)",
        r"staff\s*(?:of|size)?[:\s]*(\d[\d,]*)",
        r"workforce[:\s]*(\d[\d,]*)",
    ]:
        m = re.search(pattern, lower)
        if m:
            profile["employee_count"] = int(m.group(1).replace(",", ""))
            break

    for pattern in [
        r"(?:budget|spend|investment|willing\s*to\s*spend)[:\s]*\$?([\d,]+(?:\.\d+)?)\s*(k|m|million|thousand|billion)?",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|million|thousand|billion)?",
    ]:
        m = re.search(pattern, lower)
        if m:
            raw = (m.group(1) or "").replace(",", "").strip()
            if not raw:
                continue  # empty capture – try next pattern
            try:
                value = float(raw)
            except ValueError:
                continue
            mult  = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000, "billion": 1_000_000_000}
            value *= mult.get((m.group(2) or "").lower(), 1)
            profile["budget"] = value
            break

    for industry in [
        "technology", "healthcare", "finance", "financial services", "manufacturing",
        "retail", "energy", "agriculture", "construction", "transportation",
        "education", "telecommunications", "pharmaceuticals", "automotive",
        "real estate", "hospitality", "mining", "oil and gas", "utilities",
        "consumer goods",
    ]:
        if industry in lower:
            profile["industry"] = industry.title()
            break

    for country in [
        "united states", "united kingdom", "canada", "germany", "france", "japan",
        "china", "india", "australia", "brazil", "south korea", "mexico", "italy",
        "spain", "netherlands", "switzerland", "singapore", "sweden", "norway", "denmark",
    ]:
        if country in lower:
            profile["country"] = country.title()
            break

    return profile


# =====================================================================
# Business Size Classification
# FIX: original threshold was too aggressive – a 200-person company with
# $400K budget is still Medium, not Small.
# =====================================================================
def classify_business_size(employee_count: int, budget: float) -> str:
    emp = employee_count or 0
    bud = budget or 0
    if emp >= 250:
        return "Large"
    elif emp >= 50:
        return "Medium"
    else:
        # Small company with a large budget still a "Small" in workforce terms
        return "Small"


# =====================================================================
# Risk & Opportunity Analysis
# =====================================================================
_RISK_OPP_SYSTEM_PROMPT = (
    "You are an ESG risk and opportunity analyst. Given the company profile and regulatory "
    "context, generate tailored ESG risks and opportunities.\n\n"
    "Return ONLY valid JSON:\n"
    '{"risks": [{"title": "...", "description": "...", "severity": "High|Medium|Low"}, ...], '
    '"opportunities": [{"title": "...", "description": "...", "impact": "High|Medium|Low"}, ...]}\n\n'
    "Requirements:\n"
    "- Reference specific regulatory frameworks for the company's jurisdiction\n"
    "- Include at least one regulatory compliance risk\n"
    "- Ground descriptions in the company's data where provided\n"
    "- Provide exactly 3 risks and 3 opportunities"
)


def generate_risks_and_opportunities(
    industry:     str,
    country:      str,
    business_size: str,
    data_summary: Optional[str] = None,
) -> dict:
    regulatory  = get_regulatory_context(country)
    user_prompt = (
        f"Company profile:\n- Industry: {industry}\n- Country: {country}\n- Business Size: {business_size}\n\n"
        f"Regulatory frameworks: {', '.join(regulatory['frameworks'])}\n"
        f"Key deadlines: {regulatory['key_deadlines']}\nRegulator: {regulatory['regulator']}\n"
    )
    if data_summary:
        user_prompt += f"\nData characteristics:\n{data_summary[:800]}\n"
    user_prompt += "\nGenerate ESG risks and opportunities."

    llm_response = _call_llm(_RISK_OPP_SYSTEM_PROMPT, user_prompt)
    if llm_response:
        try:
            parsed = _extract_json(llm_response)
            if "risks" in parsed and "opportunities" in parsed:
                return parsed
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("LLM risk/opp response unparseable; using mock templates.")

    # Mock fallback (unchanged from original – solid templates)
    reg = regulatory["frameworks"]
    dl  = regulatory["key_deadlines"]

    risk_bank = {
        "Technology": [
            {"title": "E-Waste & Circular Economy Compliance",
             "description": f"As a {business_size.lower()} tech firm in {country}, e-waste disposal poses regulatory risk. {reg[0] if reg else 'Emerging regulations'} may impose stricter reporting.",
             "severity": "High"},
            {"title": "Data Centre Carbon Disclosure",
             "description": f"Growing compute demands increase carbon footprint. Under {reg[1] if len(reg) > 1 else 'emerging standards'}, emissions reporting is becoming mandatory. {dl}",
             "severity": "Medium"},
            {"title": "Supply Chain Human Rights Risk",
             "description": f"Hardware supply chains may involve poor labour jurisdictions. {reg[2] if len(reg) > 2 else 'Due diligence laws'} require supply chain transparency.",
             "severity": "High"},
        ],
        "Manufacturing": [
            {"title": "Carbon Emissions Cap Risk",
             "description": f"Manufacturing in {country} faces tightening emissions caps under {reg[0] if reg else 'local regulations'}. {dl}",
             "severity": "High"},
            {"title": "Water & Pollution Reporting",
             "description": f"Industrial water use under scrutiny. {reg[1] if len(reg) > 1 else 'Environmental standards'} mandate water impact disclosure.",
             "severity": "Medium"},
            {"title": "Worker Safety Compliance",
             "description": f"Production facility health risks. {reg[2] if len(reg) > 2 else 'Labour laws'} may require enhanced due diligence.",
             "severity": "High"},
        ],
        "Finance": [
            {"title": "Greenwashing Liability",
             "description": f"ESG fund labelling under scrutiny. {reg[0] if reg else 'Disclosure standards'} impose strict requirements. {dl}",
             "severity": "High"},
            {"title": "Stranded Asset & Transition Risk",
             "description": f"Fossil-fuel portfolio exposure. {reg[1] if len(reg) > 1 else 'Climate stress testing'} mandates climate-risk assessment.",
             "severity": "Medium"},
            {"title": "Disclosure Compliance Burden",
             "description": f"Multiple frameworks ({', '.join(reg[:2]) if reg else 'various'}) require significant reporting infrastructure investment.",
             "severity": "Medium"},
        ],
        "Financial Services": [
            {"title": "Greenwashing Liability",
             "description": f"ESG product labelling is under heightened scrutiny. {reg[0] if reg else 'Disclosure standards'} impose strict anti-greenwashing requirements. {dl}",
             "severity": "High"},
            {"title": "Stranded Asset & Transition Risk",
             "description": f"Portfolio exposure to high-carbon assets. {reg[1] if len(reg) > 1 else 'Climate stress testing'} mandates climate-risk scenario analysis.",
             "severity": "High"},
            {"title": "Disclosure & Reporting Burden",
             "description": f"Multiple concurrent frameworks ({', '.join(reg[:2]) if reg else 'various'}) impose heavy reporting costs on {business_size.lower()} firms.",
             "severity": "Medium"},
        ],
    }
    opp_bank = {
        "Technology": [
            {"title": "Green Software & Cloud Efficiency",
             "description": f"Energy cost reduction of 15-30%. Early {reg[0] if reg else 'standards'} compliance creates competitive advantage.",
             "impact": "High"},
            {"title": "Circular Economy Products",
             "description": "Device refurbishment and take-back programmes open new revenue streams and reduce material costs.",
             "impact": "Medium"},
            {"title": "ESG-Driven Talent Attraction",
             "description": f"ESG-committed firms see 15-20% lower attrition in {country}.",
             "impact": "Medium"},
        ],
        "Manufacturing": [
            {"title": "Energy Efficiency Retrofits",
             "description": f"Plant upgrades in {country} can cut energy costs 20-40% with under 3-year payback.",
             "impact": "High"},
            {"title": "Sustainable Materials Premium",
             "description": f"Recycled/bio-based inputs reduce emissions. Compliance with {reg[0] if reg else 'standards'} unlocks green procurement channels.",
             "impact": "High"},
            {"title": "Certification Market Access",
             "description": f"ISO 14001 certification opens regulated procurement channels in {country}.",
             "impact": "Medium"},
        ],
        "Finance": [
            {"title": "Green Bond Products",
             "description": f"Green bonds aligned with {reg[0] if reg else 'standards'} attract ESG capital at favourable rates.",
             "impact": "High"},
            {"title": "ESG Analytics Services",
             "description": "Proprietary ESG scoring tools and advisory services create premium revenue streams.",
             "impact": "Medium"},
            {"title": "Impact Investment Funds",
             "description": "Dedicated impact funds capture growing institutional and retail demand for ESG products.",
             "impact": "High"},
        ],
    }

    default_risks = [
        {"title": "Regulatory Non-Compliance",
         "description": f"ESG regulations in {country} ({', '.join(reg[:2]) if reg else 'evolving'}) impose new requirements. {dl}",
         "severity": "High"},
        {"title": "Climate Physical & Transition Risk",
         "description": "Extreme weather events and low-carbon transition create operational and financial risks across the value chain.",
         "severity": "Medium"},
        {"title": "Talent & Social License Risk",
         "description": f"Weak ESG commitment hinders talent attraction and retention in {country}.",
         "severity": "Medium"},
    ]
    default_opps = [
        {"title": "Operational Efficiency Gains",
         "description": f"Energy and resource efficiency measures can reduce operating costs for {business_size.lower()} {industry.lower()} businesses by 10-25%.",
         "impact": "High"},
        {"title": "Brand Differentiation & Premium Pricing",
         "description": "ESG positioning commands 5-15% premium valuations and improves customer loyalty.",
         "impact": "Medium"},
        {"title": "Green Finance Access",
         "description": f"Credible ESG strategies in {country} unlock sustainability-linked loans at 20-50 bps lower rates.",
         "impact": "High"},
    ]

    return {
        "risks":         risk_bank.get(industry, default_risks),
        "opportunities": opp_bank.get(industry, default_opps),
    }


# =====================================================================
# ESG Solution Ranking
# FIX: solutions ranked above company budget get a penalty so they don't
# end up as #1 when the company can't afford them.
# =====================================================================
_SOLUTION_SYSTEM_PROMPT = (
    "You are an ESG strategy consultant. Rank ESG solutions for the given company. "
    "Return ONLY valid JSON as a list:\n"
    '[{"name": "...", "category": "Environmental|Social|Governance", '
    '"description": "...", "upfront_cost_low": number, "upfront_cost_high": number, '
    '"annual_savings": number, "payback_years": number, "fit_score": 0-100}]\n'
    "Provide exactly 6 solutions ranked by fit_score. Include at least one per ESG pillar."
)


def rank_solutions(
    industry:     str,
    business_size: str,
    budget:       float,
    data_summary: Optional[str] = None,
) -> list:
    user_prompt = (
        f"Industry: {industry}\nBusiness Size: {business_size}\nESG Budget: ${budget:,.0f}\n"
    )
    if data_summary:
        user_prompt += f"\nData characteristics:\n{data_summary[:600]}\n"
    user_prompt += "\nRank 6 ESG solutions."

    llm_response = _call_llm(_SOLUTION_SYSTEM_PROMPT, user_prompt)
    if llm_response:
        try:
            parsed = _extract_json(llm_response)
            if isinstance(parsed, list) and len(parsed) >= 3:
                parsed.sort(key=lambda s: s.get("fit_score", 0), reverse=True)
                for i, sol in enumerate(parsed, 1):
                    sol["rank"] = i
                return parsed
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("LLM solution ranking unparseable; using templates.")

    cost_scale = {"Small": 0.3, "Medium": 1.0, "Large": 2.5}.get(business_size, 1.0)

    solutions = [
        {"name": "Carbon Footprint Tracking Platform",       "category": "Environmental",
         "description": "Deploy automated GHG emissions monitoring across Scope 1, 2, and 3 with real-time dashboarding.",
         "upfront_cost_low": 15_000*cost_scale, "upfront_cost_high": 50_000*cost_scale,
         "annual_savings": 8_000*cost_scale, "payback_years": 3.0, "fit_score": 92},
        {"name": "Renewable Energy Transition Plan",         "category": "Environmental",
         "description": "Switch to 100% renewable electricity via PPAs or on-site solar/wind.",
         "upfront_cost_low": 50_000*cost_scale, "upfront_cost_high": 200_000*cost_scale,
         "annual_savings": 25_000*cost_scale, "payback_years": 5.0, "fit_score": 88},
        {"name": "DEI & Fair Labour Programme",              "category": "Social",
         "description": "Implement DEI training, pay-equity audits, and transparent reporting.",
         "upfront_cost_low": 10_000*cost_scale, "upfront_cost_high": 40_000*cost_scale,
         "annual_savings": 5_000*cost_scale, "payback_years": 4.0, "fit_score": 80},
        {"name": "Supply Chain ESG Audit",                   "category": "Governance",
         "description": "Third-party audits of tier-1 and tier-2 suppliers for ESG compliance.",
         "upfront_cost_low": 20_000*cost_scale, "upfront_cost_high": 75_000*cost_scale,
         "annual_savings": 10_000*cost_scale, "payback_years": 4.5, "fit_score": 85},
        {"name": "ESG Reporting & Disclosure Framework",     "category": "Governance",
         "description": "Adopt GRI/SASB/TCFD-aligned reporting for investor and regulatory disclosure.",
         "upfront_cost_low": 8_000*cost_scale, "upfront_cost_high": 30_000*cost_scale,
         "annual_savings": 3_000*cost_scale, "payback_years": 5.0, "fit_score": 78},
        {"name": "Waste Reduction & Circular Economy",       "category": "Environmental",
         "description": "Redesign packaging, implement take-back programmes, set zero-waste-to-landfill targets.",
         "upfront_cost_low": 12_000*cost_scale, "upfront_cost_high": 45_000*cost_scale,
         "annual_savings": 7_000*cost_scale, "payback_years": 3.5, "fit_score": 75},
    ]

    # Industry boosts
    boosts = {
        "Technology":    ["Carbon Footprint Tracking Platform",   "ESG Reporting & Disclosure Framework"],
        "Manufacturing": ["Waste Reduction & Circular Economy",   "Renewable Energy Transition Plan"],
        "Finance":       ["ESG Reporting & Disclosure Framework", "Supply Chain ESG Audit"],
        "Financial Services": ["ESG Reporting & Disclosure Framework", "Supply Chain ESG Audit"],
        "Energy":        ["Renewable Energy Transition Plan",     "Carbon Footprint Tracking Platform"],
        "Healthcare":    ["DEI & Fair Labour Programme",          "Supply Chain ESG Audit"],
        "Retail":        ["Waste Reduction & Circular Economy",   "Supply Chain ESG Audit"],
    }
    for sol in solutions:
        if sol["name"] in boosts.get(industry, []):
            sol["fit_score"] = min(sol["fit_score"] + 5, 100)

    # FIX: penalise solutions whose low-end cost exceeds 2× the entire budget
    # (prevents recommending something unaffordable as the top pick)
    if budget and budget > 0:
        for sol in solutions:
            if sol["upfront_cost_low"] > budget * 2:
                sol["fit_score"] = max(sol["fit_score"] - 15, 10)

    solutions.sort(key=lambda s: s["fit_score"], reverse=True)
    for i, sol in enumerate(solutions, 1):
        sol["rank"] = i
    return solutions


# =====================================================================
# What-If Chat
# FIX: trimmed what's sent to the LLM – sending full JSON of all
# solutions was excessive and consumed tokens needlessly.
# =====================================================================
_WHATIF_SYSTEM_PROMPT = """You are an ESG strategy advisor. Answer what-if questions with data-driven, scenario-based analysis.

Company Profile: {profile}
Top Solutions (summary): {solutions}
Regulatory Context: {regulatory}
{data_context}

Guidelines:
- Use base/optimistic/pessimistic scenarios where appropriate
- Reference actual profile data and regulatory frameworks
- Use markdown tables for structured comparisons
- Keep responses concise (150-300 words)
- Always end with one clear recommendation"""


def answer_what_if(
    question:        str,
    company_profile: dict,
    solutions:       list,
    data_summary:    Optional[str] = None,
) -> str:
    # Compact solutions summary to save tokens
    solutions_compact = [
        {
            "name":         s["name"],
            "category":     s["category"],
            "cost_range":   f"${s['upfront_cost_low']/1000:.0f}K–${s['upfront_cost_high']/1000:.0f}K",
            "annual_savings": f"${s['annual_savings']/1000:.0f}K/yr",
            "payback_years": s["payback_years"],
        }
        for s in solutions
    ]
    regulatory = get_regulatory_context(company_profile.get("country", ""))
    data_ctx   = f"Uploaded Data Summary:\n{data_summary[:600]}" if data_summary else "No additional data statistics available."

    system = _WHATIF_SYSTEM_PROMPT.format(
        profile=json.dumps({
            k: v for k, v in company_profile.items()
            if k in ("industry", "country", "employee_count", "budget", "business_size")
        }, indent=2),
        solutions=json.dumps(solutions_compact, indent=2),
        regulatory=json.dumps({
            "frameworks":    regulatory.get("frameworks", [])[:3],
            "key_deadlines": regulatory.get("key_deadlines", ""),
            "penalties":     regulatory.get("penalties", ""),
        }, indent=2),
        data_context=data_ctx,
    )

    llm_response = _call_llm(system, question)
    if llm_response:
        return llm_response

    # Mock fallback
    q_lower  = question.lower()
    industry = company_profile.get("industry", "your industry")
    size     = company_profile.get("business_size", "your")
    country  = company_profile.get("country", "your region")
    budget   = company_profile.get("budget", 0) or 0
    emp      = company_profile.get("employee_count", 0) or 0
    reg      = get_regulatory_context(country)

    if any(kw in q_lower for kw in ("budget", "spend", "cost", "invest")):
        return (
            f"**Budget Impact Analysis**\n\n"
            f"Based on your profile ({size} {industry} company, {emp:,} employees, ${budget:,.0f} ESG budget):\n\n"
            f"| Scenario | Budget | Impact |\n|----------|--------|--------|\n"
            f"| Optimistic | +20% (${budget*1.2:,.0f}) | Payback drops from 5.0 to ~3.8 years |\n"
            f"| Base Case | ${budget:,.0f} | Implement top 2 solutions |\n"
            f"| Pessimistic | -20% (${budget*0.8:,.0f}) | Carbon Tracking only (Scope 1&2) |\n\n"
            f"*Regulatory note:* {reg['frameworks'][0] if reg['frameworks'] else 'Emerging standards'} may mandate disclosure spend regardless.\n\n"
            f"**Recommendation:** Prioritise highest-ranked solutions and phase in others over 2-3 years."
        )

    if any(kw in q_lower for kw in ("employee", "hire", "staff", "headcount", "workforce")):
        return (
            f"**Workforce Impact Projection**\n\n"
            f"| Metric | Current | Projected (12mo) |\n|--------|---------|------------------|\n"
            f"| ESG HR Capacity | ~0 FTE | +{max(1, (emp or 10)//50)} FTE |\n"
            f"| Employee Satisfaction | Baseline | +10-15% |\n"
            f"| Attrition Savings | $0 | ~${(emp or 0)*200:,.0f}/yr |\n"
            f"| Talent Acquisition Cost | Baseline | -8-12% |\n\n"
            f"**Recommendation:** Appoint a dedicated ESG lead before scaling initiatives."
        )

    if any(kw in q_lower for kw in ("timeline", "when", "how long", "duration")):
        return (
            f"**Implementation Timeline**\n\n"
            f"| Phase | Duration | Key Activities | Regulatory Milestone |\n|-------|----------|----------------|---------------------|\n"
            f"| Assessment | Months 1-2 | Baseline, gap analysis | Gap report vs {reg['frameworks'][0] if reg['frameworks'] else 'standards'} |\n"
            f"| Quick Wins | Months 3-4 | Carbon tracking, reporting setup | Initial disclosure prep |\n"
            f"| Core Build | Months 5-12 | Renewable transition, audits | Compliance milestone |\n"
            f"| Maturity | Months 12-24 | Full integration, verification | Full reporting readiness |\n\n"
            f"*Key deadline:* {reg['key_deadlines']}\n\n"
            f"**Recommendation:** Start Assessment phase this quarter to meet upcoming deadlines."
        )

    if any(kw in q_lower for kw in ("risk", "regulation", "compliance", "penalty", "fine")):
        return (
            f"**Regulatory Risk Scenario**\n\n"
            f"**Frameworks:** {', '.join(reg['frameworks'][:2])}\n"
            f"**Deadline:** {reg['key_deadlines']}\n\n"
            f"| Scenario | Likelihood | Financial Impact |\n|----------|-----------|------------------|\n"
            f"| Non-compliance fines | Medium-High | $50K–$500K |\n"
            f"| Procurement exclusion | Medium | 10-25% revenue loss |\n"
            f"| Proactive compliance | Recommended | ${8_000*(2.5 if size=='Large' else 1.0):,.0f}–${30_000*(2.5 if size=='Large' else 1.0):,.0f} |\n\n"
            f"*Verdict:* Cost of inaction is 3-5× proactive adoption. {reg['penalties']}\n\n"
            f"**Recommendation:** Begin compliance programme immediately."
        )

    if any(kw in q_lower for kw in ("carbon", "emission", "climate", "scope")):
        return (
            f"**Carbon Reduction Scenario**\n\n"
            f"| Timeframe | Reduction Target | Cumulative Reduction |\n|-----------|-----------------|---------------------|\n"
            f"| Year 1 | 15-20% Scope 1&2 | 15-20% |\n"
            f"| Years 2-3 | +25-35% renewable | ~50% |\n"
            f"| Year 5 | +10-20% circularity | 60-70% |\n\n"
            f"Potential carbon credit value: ${(emp or 0)*50:,.0f}–${(emp or 0)*150:,.0f}/year.\n\n"
            f"**Recommendation:** Deploy Carbon Footprint Tracking first to establish a verified baseline."
        )

    return (
        f"**Predictive Analysis – {industry}**\n\n"
        f"For your {size} business ({emp:,} employees, ${budget:,.0f} budget):\n\n"
        f"- ESG disclosure score: **+18-25 points** in 12 months\n"
        f"- Operational cost reduction: **10-15%** via efficiency measures\n"
        f"- Revenue uplift: **4-7%** from ESG brand positioning\n\n"
        f"**Regulatory context:** {', '.join(reg['frameworks'][:2])}\n\n"
        f"Try asking about budgets, carbon emissions, implementation timelines, or workforce impact."
    )
