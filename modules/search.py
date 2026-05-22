"""
Search module: issues queries to Tavily and returns structured results.
"""
import os
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    domain: str = ""
    published_date: Optional[str] = None

    def __post_init__(self):
        if not self.domain and self.url:
            from urllib.parse import urlparse
            self.domain = urlparse(self.url).netloc.replace("www.", "")


def search_web(query: str, max_results: int = 8, search_depth: str = "advanced") -> list[SearchResult]:
    """
    Issue a search query via Tavily API and return structured results.
    Falls back to a mock if API key not present (for testing).
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    
    if not api_key:
        return _mock_search(query, max_results)
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        results = []
        for r in response.get("results", []):
            results.append(SearchResult(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=r.get("content", r.get("snippet", "")),
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))
        return results
    except Exception as e:
        print(f"[Search] Tavily error: {e}")
        return _mock_search(query, max_results)


def multi_search(queries: list[str], max_per_query: int = 5) -> list[SearchResult]:
    """Issue multiple search queries and deduplicate results."""
    seen_urls = set()
    all_results = []
    for q in queries:
        for r in search_web(q, max_results=max_per_query):
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)
        time.sleep(0.2)  # gentle rate limiting
    return all_results


def _mock_search(query: str, max_results: int) -> list[SearchResult]:
    """Return placeholder results when no API key is configured."""
    return [
        SearchResult(
            title=f"Mock Result {i+1} for: {query[:40]}",
            url=f"https://example.com/result-{i+1}",
            snippet=f"This is a placeholder snippet for result {i+1}. Configure TAVILY_API_KEY for real results.",
            score=1.0 - (i * 0.1),
        )
        for i in range(min(max_results, 3))
    ]
