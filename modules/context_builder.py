"""
Context builder: selects relevant snippets from fetched pages, enforces token limits,
and assembles final context with citation metadata.
"""
import re
from dataclasses import dataclass
from typing import Optional

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(text) // 4


MAX_CONTEXT_TOKENS = 12_000
SNIPPET_MAX_CHARS = 2_500
MIN_SNIPPET_CHARS = 150


@dataclass
class ContextSnippet:
    url: str
    title: str
    domain: str
    text: str
    relevance_score: float = 0.0
    citation_id: int = 0


def build_context(
    query: str,
    fetched_pages,   # list[FetchedPage]
    search_results,  # list[SearchResult] for relevance scores
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[list[ContextSnippet], str]:
    """
    Select snippets from fetched pages, score by relevance, deduplicate domains,
    and return a list of ContextSnippets + formatted context string.
    """
    score_map = {r.url: r.score for r in search_results}
    query_terms = set(re.findall(r'\w+', query.lower()))

    candidates = []
    for page in fetched_pages:
        if page.fetch_error or not page.text.strip():
            continue
        snippet_text = _extract_best_snippet(page.text, query_terms)
        if len(snippet_text) < MIN_SNIPPET_CHARS:
            continue
        relevance = score_map.get(page.url, 0.0)
        relevance += _keyword_score(snippet_text, query_terms)
        candidates.append(ContextSnippet(
            url=page.url,
            title=page.title or page.domain,
            domain=page.domain,
            text=snippet_text,
            relevance_score=relevance,
        ))

    # Sort by relevance, deduplicate domains (keep best per domain)
    candidates.sort(key=lambda x: x.relevance_score, reverse=True)
    seen_domains = {}
    deduped = []
    for c in candidates:
        if c.domain not in seen_domains:
            seen_domains[c.domain] = c
            deduped.append(c)
        elif c.relevance_score > seen_domains[c.domain].relevance_score * 1.2:
            seen_domains[c.domain] = c
            deduped = [x for x in deduped if x.domain != c.domain] + [c]

    # Assign citation IDs and trim to token budget
    token_budget = max_tokens
    selected = []
    for i, snippet in enumerate(deduped, 1):
        tokens = count_tokens(snippet.text)
        if tokens > token_budget:
            # Truncate snippet to fit
            ratio = token_budget / tokens
            snippet.text = snippet.text[:int(len(snippet.text) * ratio)]
            tokens = count_tokens(snippet.text)
        if tokens > token_budget:
            break
        snippet.citation_id = i
        selected.append(snippet)
        token_budget -= tokens
        if token_budget < 200:
            break

    context_str = _format_context(selected)
    return selected, context_str


def _extract_best_snippet(text: str, query_terms: set[str]) -> str:
    """Find the most relevant portion of a page's text."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 80]
    if not paragraphs:
        return text[:SNIPPET_MAX_CHARS]

    # Score each paragraph by query term overlap
    scored = []
    for p in paragraphs:
        p_words = set(re.findall(r'\w+', p.lower()))
        score = len(query_terms & p_words) / max(len(query_terms), 1)
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    # Take top paragraphs up to SNIPPET_MAX_CHARS
    result = []
    total = 0
    for score, para in scored:
        if total + len(para) > SNIPPET_MAX_CHARS:
            break
        result.append(para)
        total += len(para) + 2

    return "\n\n".join(result) if result else text[:SNIPPET_MAX_CHARS]


def _keyword_score(text: str, query_terms: set[str]) -> float:
    """Return a 0–1 score based on query term coverage in text."""
    words = set(re.findall(r'\w+', text.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & words) / len(query_terms)


def _format_context(snippets: list[ContextSnippet]) -> str:
    """Format snippets into an LLM-ready context block."""
    parts = []
    for s in snippets:
        parts.append(
            f"[SOURCE {s.citation_id}] {s.title}\n"
            f"URL: {s.url}\n"
            f"Domain: {s.domain}\n"
            f"---\n{s.text}\n"
        )
    return "\n\n".join(parts)


def build_summary_prompt(conversation_history: list[dict], max_tokens: int = 800) -> str:
    """
    Summarize older conversation turns into a compact rolling summary
    when history exceeds context limits.
    """
    turns_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in conversation_history[-10:]
    )
    return f"[CONVERSATION SUMMARY]\n{turns_text}\n[END SUMMARY]"
