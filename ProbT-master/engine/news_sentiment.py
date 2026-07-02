"""News fetch + sentiment scoring.

Sources:
  - Headlines: NewsAPI (external; key in env / Secret Manager on GCP).
  - Sentiment: Vertex AI Gemini via google-genai unified SDK.
    Uses ADC on Cloud Run / `gcloud auth application-default login`,
    or GEMINI_API_KEY for local dev.

Graceful degradation: if any API is unavailable, returns neutral 0.0 and
logs a note. Never raises — the pipeline must keep running.

Exports: fetch_headlines, score_sentiment.
"""
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = (os.getenv("NEWSAPI_KEY") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GCLOUD_PROJECT = (os.getenv("GCLOUD_PROJECT") or "").strip()
GCLOUD_LOCATION = (os.getenv("GCLOUD_LOCATION") or "us-central1").strip()

_GEMINI_MODEL = "gemini-2.5-flash"

# ─── TTL cache for news (respects NewsAPI free plan: 100 req/day) ──────
# With 6h TTL → 4 req/day per unique query. Even with 5 symbols × 3 Cloud
# Run instances that's ≤60 req/day, comfortably under the 100 limit.
_NEWS_CACHE: dict = {}  # query -> (timestamp, result)
_NEWS_TTL = 6 * 3600    # 6 hours in seconds


def fetch_headlines(
    query: str = 'gold OR XAUUSD OR "gold price"',
    hours: int = 72,
    page_size: int = 20,
) -> List[dict]:
    """Fetch recent gold-related headlines from NewsAPI.

    Default `hours=72` because the NewsAPI free/developer plan has a 48-hour
    embargo on /v2/everything — querying within the last 24h returns 0 results.

    Returns list of {title, published_at, source}. Empty list on any error
    or if NEWSAPI_KEY is not set.
    """
    if not NEWSAPI_KEY:
        return []
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": since,
                "sortBy": "relevancy",
                "pageSize": page_size,
                "language": "en",
                "apiKey": NEWSAPI_KEY,
            },
            timeout=15,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "published_at": a.get("publishedAt"),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception:
        return []


def _build_gemini_client():
    """Lazy-import and build a Gemini client. Returns None if unavailable."""
    try:
        from google import genai
    except Exception as e:
        print(f"[news] google.genai import failed: {e}", flush=True)
        return None
    try:
        if GEMINI_API_KEY:
            print("[news] using Gemini API key path", flush=True)
            return genai.Client(api_key=GEMINI_API_KEY)
        if GCLOUD_PROJECT:
            print(f"[news] using Vertex AI ADC path (project={GCLOUD_PROJECT})", flush=True)
            return genai.Client(
                vertexai=True, project=GCLOUD_PROJECT, location=GCLOUD_LOCATION
            )
        # Fallback: let the SDK auto-discover ADC
        print("[news] using Vertex AI ADC fallback (no project)", flush=True)
        return genai.Client(vertexai=True, project="", location=GCLOUD_LOCATION)
    except Exception as e:
        print(f"[news] gemini client build failed: {e}", flush=True)
        return None


def _gemini_score(text: str) -> tuple[float, str]:
    """Score aggregate sentiment in [-1, 1] via Gemini. Returns (score, debug_msg)."""
    prompt = (
        "You are a financial sentiment analyst for XAUUSD (gold). Score the "
        "aggregate sentiment of the following news headlines toward GOLD price. "
        "Return ONLY a single decimal number in [-1, 1] where -1=very bearish, "
        "0=neutral, +1=very bullish.\n\nHeadlines:\n" + text
    )
    client = _build_gemini_client()
    if client is None:
        return 0.0, "client=None"
    try:
        resp = client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
        raw = (resp.text or "").strip()
        print(f"[news] gemini raw response: {raw!r}", flush=True)
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if m:
            score = max(-1.0, min(1.0, float(m.group(0))))
            print(f"[news] gemini score: {score}", flush=True)
            return score, "ok"
        return 0.0, f"no-number-in-response: {raw!r}"
    except Exception as e:
        print(f"[news] gemini generate_content failed: {e}", flush=True)
        return 0.0, f"exception: {e}"


def score_sentiment(headlines: List[dict]) -> dict:
    """Returns {score, count, summary}.

    score in [-1, 1]. 0.0 if no headlines or API unavailable.
    """
    if not headlines:
        return {
            "score": 0.0,
            "count": 0,
            "summary": "No headlines available (NewsAPI key not configured or no results).",
        }
    joined = "\n".join(f"- {h['title']}" for h in headlines[:20])
    score, _debug = _gemini_score(joined)
    tag = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    return {
        "score": round(score, 4),
        "count": len(headlines),
        "summary": f"News sentiment {tag} ({score:+.2f}) from {len(headlines)} headlines.",
    }


def get_news_sentiment(
    query: str = 'gold OR XAUUSD OR "gold price"',
    hours: int = 72,
) -> dict:
    """Cached fetch_headlines + score_sentiment with a 6h TTL.

    This is the entry point the live engine should call — it avoids hitting
    NewsAPI (100 req/day free limit) and Gemini on every 60s reading refresh.
    """
    now = time.time()
    cached = _NEWS_CACHE.get(query)
    if cached and (now - cached[0]) < _NEWS_TTL:
        return cached[1]
    headlines = fetch_headlines(query=query, hours=hours)
    result = score_sentiment(headlines)
    _NEWS_CACHE[query] = (now, result)
    return result
