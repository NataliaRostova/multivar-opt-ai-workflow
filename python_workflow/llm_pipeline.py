"""Unified DeepSeek / Gemini API Structured Financial Information Extraction Pipeline.

Supports:
- Live DeepSeek API (`DEEPSEEK_API_KEY` env var, OpenAI-compatible chat completions endpoint)
- Live Gemini API (`GEMINI_API_KEY` env var, Google Generative Language REST API)
- Deterministic offline `demo` fallback when no API key is present so the workflow is 100% reproducible.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any


SYSTEM_PROMPT = (
    "You are a quantitative financial research analyst. Extract strict JSON with keys: "
    "entity, period, revenue_bn_rmb, revenue_yoy_pct, net_profit_bn_rmb, sentiment "
    "(Positive/Neutral/Negative), key_catalyst, and risk_flag. Return raw JSON only."
)


@dataclass
class StructuredFiling:
    doc_id: str
    entity: str
    period: str
    revenue_bn_rmb: float | None
    revenue_yoy_pct: float | None
    net_profit_bn_rmb: float | None
    sentiment: str
    key_catalyst: str
    risk_flag: str
    provider: str


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_deepseek_api(text: str, api_key: str, model: str = "deepseek-chat") -> dict[str, Any]:
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    res = _http_post_json(url, payload, headers)
    raw_content = res["choices"][0]["message"]["content"]
    return json.loads(raw_content)


def call_gemini_api(text: str, api_key: str, model: str = "gemini-2.5-flash") -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nDocument:\n{text}"}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    headers = {"Content-Type": "application/json"}
    res = _http_post_json(url, payload, headers)
    raw_content = res["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(raw_content)


def offline_rule_extractor(doc: dict[str, str]) -> dict[str, Any]:
    """Deterministic structured fallback parser for offline reproducibility without API keys."""
    text = doc["text"]
    rev_m = re.search(r"营业收入(?:约)?\s*([\d.]+)\s*亿元", text)
    yoy_m = re.search(r"同比增长\s*([+-]?[\d.]+)%", text)
    np_m = re.search(r"归母净利润(?:约)?\s*([\d.]+)\s*亿元", text)

    yoy = float(yoy_m.group(1)) if yoy_m else None
    sentiment = "Positive" if (yoy is not None and yoy >= 8.0) else ("Negative" if (yoy is not None and yoy < 0) else "Neutral")

    return {
        "entity": doc["entity_hint"],
        "period": doc["period_hint"],
        "revenue_bn_rmb": float(rev_m.group(1)) if rev_m else None,
        "revenue_yoy_pct": yoy,
        "net_profit_bn_rmb": float(np_m.group(1)) if np_m else None,
        "sentiment": sentiment,
        "key_catalyst": doc["catalyst_hint"],
        "risk_flag": doc["risk_hint"],
    }


def run_structured_extraction(docs: list[dict[str, str]], prefer_provider: str = "auto") -> list[dict[str, Any]]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    results: list[dict[str, Any]] = []
    for doc in docs:
        provider_used = "offline-schema-parser"
        parsed: dict[str, Any] | None = None

        if prefer_provider in ("auto", "deepseek") and deepseek_key:
            try:
                parsed = call_deepseek_api(doc["text"], deepseek_key)
                provider_used = "deepseek-api"
            except Exception:
                parsed = None

        if parsed is None and prefer_provider in ("auto", "gemini") and gemini_key:
            try:
                parsed = call_gemini_api(doc["text"], gemini_key)
                provider_used = "gemini-api"
            except Exception:
                parsed = None

        if parsed is None:
            parsed = offline_rule_extractor(doc)

        record = StructuredFiling(
            doc_id=doc["doc_id"],
            entity=str(parsed.get("entity", doc["entity_hint"])),
            period=str(parsed.get("period", doc["period_hint"])),
            revenue_bn_rmb=parsed.get("revenue_bn_rmb"),
            revenue_yoy_pct=parsed.get("revenue_yoy_pct"),
            net_profit_bn_rmb=parsed.get("net_profit_bn_rmb"),
            sentiment=str(parsed.get("sentiment", "Neutral")),
            key_catalyst=str(parsed.get("key_catalyst", "")),
            risk_flag=str(parsed.get("risk_flag", "")),
            provider=provider_used,
        )
        results.append(asdict(record))
    return results
