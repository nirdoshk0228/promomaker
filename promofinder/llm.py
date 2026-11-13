"""Optional LLM judge: re-scores the heuristic shortlist the way a promo editor would.

Heuristics are fast and free but cannot tell a side-plot outburst from a story-defining moment.
The LLM only sees the ~30 shortlisted clips (not the whole show), so a full run costs a few cents.

Configuration (environment variables):
  PROMOFINDER_LLM_PROVIDER  openai | groq | gemini | openrouter | anthropic | custom
  PROMOFINDER_LLM_API_KEY   API key for that provider
  PROMOFINDER_LLM_MODEL     model name (sensible default per provider)
  PROMOFINDER_LLM_BASE_URL  only for provider=custom (any OpenAI-compatible endpoint)
  PROMOFINDER_LLM_WEIGHT    blend weight of the LLM score (default 0.5)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import httpx

from .features import EpisodeIndex
from .models import Candidate

OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"),
}
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You are a senior promo editor for a vertical micro-drama app (like ReelShort, DramaBox, Kuku TV).
You pick 20-60 second clips from shows to post on Instagram/YouTube/TikTok to pull in NEW viewers.

A great promo:
1. HOOK: something in the first 3 seconds makes a scrolling stranger stop (a slap, a reveal, a shocking line).
2. CLIFFHANGER: it cuts at an unresolved moment so the viewer needs to know what happens next.
3. STANDALONE: someone who has never seen the show understands who is fighting whom and why it matters.
Penalise clips that start mid-conversation, depend on earlier episodes, are exposition (money, legal, logistics),
are side-plots, dreams, comic relief, or end on a resolved/happy beat.

Dialogue may be in Hinglish (Hindi in Roman script). Judge each clip on its own. Be strict: most clips are not 9s."""

USER_TEMPLATE = """Score each clip. Timestamps are relative to the start of the clip; "||" marks a cut between moments.

{clips}

Reply with ONLY a JSON array, one object per clip, in this exact shape:
[{{"id": "<clip id>", "hook": 0-10, "cliffhanger": 0-10, "standalone": 0-10, "verdict": "use"|"maybe"|"skip", "reason": "<max 25 words>"}}]"""


class LLMError(RuntimeError):
    pass


class LLMFatal(LLMError):
    """An error retrying will not fix (bad key, unknown model, request too large, daily quota used up)."""


def _api_error_message(r: httpx.Response) -> str:
    try:
        data = r.json()
    except ValueError:
        return (r.text or "").strip()[:200]
    if isinstance(data, list) and data:
        data = data[0]
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        err = err.get("message") or err.get("type") or json.dumps(err)[:200]
    return str(err).strip()[:200]


def _retry_after(r: httpx.Response, message: str) -> Optional[float]:
    """Seconds to wait before retrying a 429, from the Retry-After header or the provider's message."""
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)(ms|s)", message)
    if m:
        mins, val, unit = m.groups()
        secs = float(val) / (1000 if unit == "ms" else 1)
        return secs + 60 * int(mins or 0)
    m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", message, re.I)  # Gemini: "Please retry in 21.4s"
    return float(m.group(1)) if m else None


def clip_transcript(c: Candidate, idxs: dict[int, EpisodeIndex], max_lines: int = 30, max_chars: int = 160) -> str:
    """Compact transcript (keeps token use low enough for free-tier rate limits)."""
    lines, offset = [], 0.0
    for n, seg in enumerate(c.segments):
        if n:
            lines.append(f"|| cut to Episode {seg.episode}")
        idx = idxs[seg.episode]
        for b in idx.blocks_between(seg.start, seg.end):
            t = offset + b.start - seg.start
            if b.kind == "DIALOGUE":
                emo = f" ({', '.join(b.emotions + b.modifiers)})" if (b.emotions or b.modifiers) else ""
                lines.append(f"[+{t:4.1f}s] {b.speaker}{emo}: \"{b.text[:max_chars]}\"")
            else:
                lines.append(f"[+{t:4.1f}s] {b.kind}: {b.text[:max_chars]}")
        offset += seg.duration
    if len(lines) > max_lines:
        lines = lines[: max_lines - 6] + ["[...]"] + lines[-5:]
    return "\n".join(lines)


def _cid(c: Candidate) -> str:
    return "-".join(f"e{s.episode}_{s.start:.1f}_{s.end:.1f}" for s in c.segments)


def _extract_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise LLMError("model reply contained no JSON array")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise LLMError("model reply was not a list")
    return data


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 60.0, transport: Optional[httpx.BaseTransport] = None, max_wait: float = 30.0,
                 max_attempts: int = 4):
        self.provider = provider.lower()
        if self.provider == "anthropic":
            self.base_url = base_url or "https://api.anthropic.com/v1"
            self.model = model or ANTHROPIC_DEFAULT_MODEL
        elif self.provider in OPENAI_COMPAT:
            self.base_url, default_model = OPENAI_COMPAT[self.provider]
            self.base_url = base_url or self.base_url
            self.model = model or default_model
        elif self.provider == "custom":
            if not base_url or not model:
                raise LLMError("provider=custom needs PROMOFINDER_LLM_BASE_URL and PROMOFINDER_LLM_MODEL")
            self.base_url, self.model = base_url, model
        else:
            raise LLMError(f"unknown provider '{provider}'")
        self.api_key = api_key
        self.max_wait, self.max_attempts = max_wait, max_attempts
        self.http = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, system: str, user: str) -> str:
        """One completion with retries. Rate limits (429) wait as long as the provider asks, up to max_wait."""
        last = "unknown error"
        for attempt in range(self.max_attempts):
            try:
                return self._complete_once(system, user)
            except httpx.HTTPStatusError as exc:
                r = exc.response
                msg = _api_error_message(r)
                last = f"HTTP {r.status_code}: {msg}"
                if r.status_code == 429:
                    wait = _retry_after(r, msg)
                    if wait is None:
                        wait = 5.0 * (attempt + 1)
                    if wait > self.max_wait or re.search(r"per day|daily|quota", msg, re.I) and wait > 5:
                        raise LLMFatal(f"{self.provider}/{self.model} rate limit — {last}")
                    time.sleep(wait + 0.5)
                    continue
                if r.status_code in (400, 401, 403, 404, 413, 422):
                    raise LLMFatal(f"{self.provider}/{self.model} — {last}")
                time.sleep(2.0 * (attempt + 1))
            except httpx.TransportError as exc:
                last = f"network error: {exc}"
                time.sleep(2.0 * (attempt + 1))
            except (KeyError, IndexError, ValueError) as exc:
                last = f"unexpected reply: {exc}"
                time.sleep(1.0)
        raise LLMError(f"{self.provider}/{self.model} — {last}")

    def _complete_once(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            r = self.http.post(f"{self.base_url}/messages", headers={
                "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": self.model, "max_tokens": 2000, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            return "".join(p.get("text", "") for p in r.json().get("content", []))
        r = self.http.post(f"{self.base_url}/chat/completions", headers={
            "Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class Reranker:
    """Callable used by pipeline.analyze: rescoring happens in place on the candidates."""

    def __init__(self, client: LLMClient, weight: float = 0.5, batch_size: int = 6, workers: int = 1,
                 cache: Optional[dict] = None, max_clips: int = 60):
        self.client, self.weight, self.batch_size, self.workers = client, weight, batch_size, workers
        self.max_clips = max_clips          # pipeline sends at most this many clips per run
        self.cache = cache if cache is not None else {}

    def _key(self, transcript: str) -> str:
        return hashlib.sha256(f"{self.client.model}\n{transcript}".encode()).hexdigest()

    def _judge_batch(self, batch: list[tuple[str, str]]) -> dict:
        clips = "\n\n".join(f"### Clip id: {cid}\n{tr}" for cid, tr in batch)
        reply = self.client.complete(SYSTEM_PROMPT, USER_TEMPLATE.format(clips=clips))
        out = {}
        for item in _extract_json_array(reply):
            try:
                out[str(item["id"])] = {
                    "hook": float(item["hook"]), "cliffhanger": float(item["cliffhanger"]),
                    "standalone": float(item["standalone"]), "verdict": str(item.get("verdict", "maybe")),
                    "reason": str(item.get("reason", ""))[:240]}
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def __call__(self, cands: list[Candidate], idxs: dict[int, EpisodeIndex]) -> dict:
        t0 = time.perf_counter()
        todo, judged = [], {}
        for c in cands:
            tr = clip_transcript(c, idxs)
            k = self._key(tr)
            if k in self.cache:
                judged[_cid(c)] = self.cache[k]
            else:
                todo.append((c, tr, k))
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        errors = []
        stop = {"flag": False}

        def run(batch):
            if stop["flag"]:
                return batch, {}, None
            try:
                return batch, self._judge_batch([(_cid(c), tr) for c, tr, _ in batch]), None
            except LLMFatal as exc:   # no point sending more batches
                stop["flag"] = True
                return batch, {}, str(exc)
            except Exception as exc:  # keep going: a failed batch just keeps heuristic scores
                return batch, {}, str(exc)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for batch, res, err in pool.map(run, batches):
                if err:
                    errors.append(err)
                for c, _, k in batch:
                    if _cid(c) in res:
                        self.cache[k] = res[_cid(c)]
                        judged[_cid(c)] = res[_cid(c)]
        n = 0
        for c in cands:
            j = judged.get(_cid(c))
            if not j:
                continue
            llm_score = max(0.0, min(1.0, (0.35 * j["hook"] + 0.35 * j["cliffhanger"] + 0.3 * j["standalone"]) / 10))
            if j["verdict"] == "skip":
                llm_score *= 0.7
            c.llm = {**j, "score": round(llm_score, 4), "heuristic_score": round(c.score, 4)}
            c.score = (1 - self.weight) * c.score + self.weight * llm_score
            if j["reason"]:
                c.reasons.insert(0, f"Editor AI: {j['reason']}")
            n += 1
        return {"provider": self.client.provider, "model": self.client.model, "judged": n, "shortlist": len(cands),
                "cached": len(cands) - len(todo), "errors": list(dict.fromkeys(errors))[:3],
                "stopped_early": stop["flag"], "seconds": round(time.perf_counter() - t0, 2), "weight": self.weight}


def make_reranker_from_env(cache: Optional[dict] = None) -> Optional[Reranker]:
    provider = os.getenv("PROMOFINDER_LLM_PROVIDER", "").strip()
    key = os.getenv("PROMOFINDER_LLM_API_KEY", "").strip()
    if not provider or not key:
        return None
    client = LLMClient(provider, key, model=os.getenv("PROMOFINDER_LLM_MODEL") or None,
                       base_url=os.getenv("PROMOFINDER_LLM_BASE_URL") or None,
                       max_wait=float(os.getenv("PROMOFINDER_LLM_MAX_WAIT", "30")))
    # free tiers allow only a few requests/tokens per minute: go one batch at a time by default
    return Reranker(client, weight=float(os.getenv("PROMOFINDER_LLM_WEIGHT", "0.5")), cache=cache,
                    batch_size=int(os.getenv("PROMOFINDER_LLM_BATCH", "6")),
                    workers=int(os.getenv("PROMOFINDER_LLM_WORKERS", "1")),
                    max_clips=int(os.getenv("PROMOFINDER_LLM_MAX_CLIPS", "60")))


def llm_status() -> dict:
    provider = os.getenv("PROMOFINDER_LLM_PROVIDER", "").strip()
    has_key = bool(os.getenv("PROMOFINDER_LLM_API_KEY", "").strip())
    model = os.getenv("PROMOFINDER_LLM_MODEL") or (
        ANTHROPIC_DEFAULT_MODEL if provider == "anthropic" else OPENAI_COMPAT.get(provider, (None, None))[1])
    return {"available": bool(provider and has_key), "provider": provider or None, "model": model}
