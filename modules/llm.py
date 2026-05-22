import os
from typing import Generator
from groq import Groq

MODEL = "llama-3.3-70b-versatile"
client = None

def _get_client():
    global client
    if client is None:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return client


PLAN_SYSTEM = """You are a research planning assistant. Given a user query and conversation context,
produce a concise research plan with:
1. A one-sentence problem framing
2. 2-4 specific search queries to issue (each on a new line starting with QUERY:)
3. What to look for in results
"""

ANSWER_SYSTEM = """You are a Deep Research Agent. Answer only from provided sources.
Cite sources using [N] notation.
"""

SUMMARIZE_SYSTEM = """Summarize the conversation in 2-3 sentences."""


def _chat(system_prompt: str, user_prompt: str, max_tokens: int = 1000):
    response = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def plan_research(query: str, conversation_summary: str = ""):
    context = f"Conversation so far: {conversation_summary}\n\n" if conversation_summary else ""
    prompt = f"{context}User query: {query}"

    plan_text = _chat(PLAN_SYSTEM, prompt, 500)

    search_queries = []
    for line in plan_text.splitlines():
        if line.strip().startswith("QUERY:"):
            search_queries.append(line.split("QUERY:", 1)[1].strip())

    if not search_queries:
        search_queries = [query]

    return plan_text, search_queries


def generate_answer_stream(
    query,
    context_str,
    conversation_history,
    conversation_summary=""
):
    prompt = f"""
Research Context:

{context_str}

User Question:
{query}

Answer using only the provided sources.
"""

    answer = _chat(ANSWER_SYSTEM, prompt, 2000)

    for chunk in answer.split():
        yield chunk + " "


def generate_answer_sync(
    query,
    context_str,
    conversation_history,
    conversation_summary=""
):
    return "".join(
        generate_answer_stream(
            query,
            context_str,
            conversation_history,
            conversation_summary
        )
    )


def summarize_conversation(conversation_history):
    history_text = "\n".join(
        f"{m['role']}: {m['content'][:400]}"
        for m in conversation_history
    )

    return _chat(SUMMARIZE_SYSTEM, history_text, 200)