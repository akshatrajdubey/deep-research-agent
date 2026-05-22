"""
Agent orchestration: the main research loop.
Produces streaming status updates and a final answer.
Builds without any agent framework (pure Python).
"""
from dataclasses import dataclass, field
from typing import Generator, Optional
from modules.search import search_web, multi_search, SearchResult
from modules.fetcher import fetch_pages_batch, FetchedPage
from modules.context_builder import build_context, ContextSnippet
from modules.session import (
    add_message, save_turn, get_messages, get_turns,
    update_session_title,
)
from modules import llm, session as session_mod


@dataclass
class AgentStep:
    """Represents one streaming progress update."""
    stage: str           # planning | searching | fetching | selecting | answering | done | error
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    answer: str
    search_queries: list[str]
    search_results: list[SearchResult]
    fetched_pages: list[FetchedPage]
    snippets: list[ContextSnippet]
    plan_text: str


def run_research_turn(
    session_id: str,
    user_query: str,
) -> Generator[AgentStep | str, None, None]:
    """
    Main agent loop. Yields AgentStep objects for progress updates,
    then yields strings (answer tokens) once answering begins.

    Usage:
        for item in run_research_turn(session_id, query):
            if isinstance(item, AgentStep):
                handle_status(item)
            else:
                stream_to_ui(item)
    """
    # ─── Persist user message ────────────────────────────────────────────────
    add_message(session_id, "user", user_query)
    conversation_history = get_messages(session_id)

    # Rolling summary if history is long
    conversation_summary = ""
    if len(conversation_history) > 8:
        yield AgentStep("summarizing", "📋 Compressing conversation history...")
        conversation_summary = llm.summarize_conversation(conversation_history)

    # ─── STAGE 1: Plan ───────────────────────────────────────────────────────
    yield AgentStep("planning", "🧠 Planning research strategy...")
    try:
        plan_text, search_queries = llm.plan_research(user_query, conversation_summary)
    except Exception as e:
        yield AgentStep("error", f"Planning failed: {e}")
        return

    yield AgentStep("planning", f"📋 Research plan ready", {
        "plan": plan_text,
        "queries": search_queries,
    })

    # ─── STAGE 2: Search ─────────────────────────────────────────────────────
    yield AgentStep("searching", f"🔍 Searching the web ({len(search_queries)} queries)...")
    try:
        search_results = multi_search(search_queries, max_per_query=5)
    except Exception as e:
        yield AgentStep("error", f"Search failed: {e}")
        return

    yield AgentStep("searching", f"✅ Found {len(search_results)} results", {
        "result_count": len(search_results),
        "results": [{"title": r.title, "url": r.url, "score": r.score} for r in search_results[:8]],
    })

    # ─── STAGE 3: Fetch pages ────────────────────────────────────────────────
    # Prioritize higher-scored results
    sorted_results = sorted(search_results, key=lambda r: r.score, reverse=True)
    urls_to_fetch = [r.url for r in sorted_results[:6]]

    yield AgentStep("fetching", f"📄 Fetching {len(urls_to_fetch)} sources...")
    try:
        fetched_pages = fetch_pages_batch(urls_to_fetch, max_pages=6)
    except Exception as e:
        yield AgentStep("error", f"Fetch failed: {e}")
        return

    successful = [p for p in fetched_pages if not p.fetch_error and p.text]
    yield AgentStep("fetching", f"✅ Retrieved content from {len(successful)}/{len(urls_to_fetch)} sources", {
        "pages": [{"url": p.url, "title": p.title, "chars": p.char_count, "error": p.fetch_error}
                  for p in fetched_pages]
    })

    # ─── STAGE 4: Select context ─────────────────────────────────────────────
    yield AgentStep("selecting", "🎯 Selecting most relevant context...")
    try:
        snippets, context_str = build_context(user_query, fetched_pages, search_results)
    except Exception as e:
        yield AgentStep("error", f"Context selection failed: {e}")
        return

    yield AgentStep("selecting", f"✅ Selected {len(snippets)} context snippets", {
        "snippets": [{"id": s.citation_id, "title": s.title, "domain": s.domain, "chars": len(s.text)}
                     for s in snippets]
    })

    # ─── STAGE 5: Generate answer ─────────────────────────────────────────────
    yield AgentStep("answering", "✍️ Generating answer with citations...")

    full_answer = ""
    try:
        for chunk in llm.generate_answer_stream(
            query=user_query,
            context_str=context_str,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        ):
            full_answer += chunk
            yield chunk  # stream tokens to UI
    except Exception as e:
        yield AgentStep("error", f"Answer generation failed: {e}")
        return

    # ─── Persist turn and assistant message ──────────────────────────────────
    add_message(session_id, "assistant", full_answer)

    # Auto-title session from first query
    session_info = session_mod.get_session(session_id)
    if session_info and session_info.get("title") == "New Session":
        short_title = user_query[:50] + ("..." if len(user_query) > 50 else "")
        update_session_title(session_id, short_title)

    save_turn(
        session_id=session_id,
        query=user_query,
        search_queries=search_queries,
        urls_opened=urls_to_fetch,
        context_snippets=[
            {"id": s.citation_id, "title": s.title, "url": s.url, "domain": s.domain}
            for s in snippets
        ],
        final_answer=full_answer,
    )

    yield AgentStep("done", "✅ Research complete", {
        "answer_length": len(full_answer),
        "sources_used": len(snippets),
    })
