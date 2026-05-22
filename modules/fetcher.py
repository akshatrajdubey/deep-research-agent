"""
Page fetch module: retrieves full page content and extracts readable text.
"""
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import requests
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
FETCH_TIMEOUT = 10
MAX_CONTENT_CHARS = 20_000


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    domain: str
    retrieved_at: str
    char_count: int = 0
    fetch_error: Optional[str] = None

    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.domain:
            self.domain = urlparse(self.url).netloc.replace("www.", "")


def fetch_page(url: str) -> FetchedPage:
    """Fetch a URL and return clean readable text."""
    domain = urlparse(url).netloc.replace("www.", "")
    retrieved_at = datetime.now(timezone.utc).isoformat()

    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return FetchedPage(
                url=url, title="", text="", domain=domain,
                retrieved_at=retrieved_at,
                fetch_error=f"Non-HTML content type: {content_type}"
            )

        html = resp.text
        title, text = _extract_text(html, url)
        text = text[:MAX_CONTENT_CHARS]
        return FetchedPage(url=url, title=title, text=text, domain=domain, retrieved_at=retrieved_at)

    except requests.exceptions.Timeout:
        return FetchedPage(url=url, title="", text="", domain=domain,
                           retrieved_at=retrieved_at, fetch_error="Timeout")
    except requests.exceptions.ConnectionError as e:
        return FetchedPage(url=url, title="", text="", domain=domain,
                           retrieved_at=retrieved_at, fetch_error=f"Connection error: {str(e)[:80]}")
    except Exception as e:
        return FetchedPage(url=url, title="", text="", domain=domain,
                           retrieved_at=retrieved_at, fetch_error=str(e)[:120])


def _extract_text(html: str, url: str) -> tuple[str, str]:
    """Extract title and readable text from HTML, preferring trafilatura."""
    title = ""
    text = ""

    # Try trafilatura first (best article extractor)
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        if extracted and len(extracted) > 200:
            text = extracted
    except Exception:
        pass

    # Extract title with BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("title")
        if tag:
            title = tag.get_text(strip=True)

        # Fallback text extraction if trafilatura failed
        if not text:
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse excessive blank lines
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            text = "\n".join(lines)
    except Exception:
        pass

    return title, text


def fetch_pages_batch(urls: list[str], max_pages: int = 6) -> list[FetchedPage]:
    """Fetch multiple pages with a small delay between requests."""
    pages = []
    for url in urls[:max_pages]:
        page = fetch_page(url)
        pages.append(page)
        time.sleep(0.3)
    return pages
