"""
Deep Research Agent — Streamlit UI
Run: python -m streamlit run app.py
"""
import sys
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from modules.config import load_env  # noqa
from modules.session import (
    init_db, create_session, list_sessions, get_messages,
    get_turns, delete_session,
)
from modules.agent import run_research_turn, AgentStep

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
.user-msg {
    background: #1a2035; border: 1px solid #2a3450;
    border-radius: 12px 12px 4px 12px; padding: 1rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;
}
.assistant-msg {
    background: #0f1520; border: 1px solid #1e2d48; border-left: 3px solid #3b82f6;
    border-radius: 4px 12px 12px 12px; padding: 1rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;
}
.stButton > button {
    background: #1d3a6b !important; border: 1px solid #3b82f6 !important;
    color: #93c5fd !important; border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Sessions")

    if st.button("➕ New Session", use_container_width=True):
        st.session_state.session_id = create_session()
        st.rerun()

    st.divider()

    sessions = list_sessions()
    if not sessions:
        st.caption("No sessions yet. Create one above.")
    else:
        for s in sessions:
            col1, col2 = st.columns([5, 1])
            with col1:
                label = s.get("title", "Session")[:35]
                if st.button(label, key=f"s_{s['session_id']}", use_container_width=True):
                    st.session_state.session_id = s["session_id"]
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"d_{s['session_id']}"):
                    delete_session(s["session_id"])
                    if st.session_state.session_id == s["session_id"]:
                        st.session_state.session_id = None
                    st.rerun()

    # Show turn history
    if st.session_state.session_id:
        turns = get_turns(st.session_state.session_id)
        if turns:
            st.divider()
            with st.expander(f"📊 Turn History ({len(turns)})"):
                for i, t in enumerate(turns, 1):
                    st.markdown(f"**Turn {i}:** {t['query'][:50]}")
                    st.caption(f"Sources: {len(t['urls_opened'])}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔬 Deep Research Agent")

if st.session_state.session_id is None:
    st.info("👈 Create a new session in the sidebar to start researching.")
    st.stop()

# Show chat history
messages = get_messages(st.session_state.session_id)
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    ts = msg.get("timestamp", "")[:16].replace("T", " ")
    if role == "user":
        st.markdown(f'<div class="user-msg"><small style="color:#475569">YOU · {ts}</small><br><br>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg"><small style="color:#1d4ed8">AGENT · {ts}</small><br><br>{content}</div>', unsafe_allow_html=True)

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_input(
    "Ask a research question",
    placeholder="e.g. What are the latest breakthroughs in quantum computing?",
    key="query_input"
)

if st.button("🔍 Search & Research", use_container_width=True):
    if not user_input.strip():
        st.warning("Please type a question first.")
    else:
        query = user_input.strip()
        st.markdown(f'<div class="user-msg"><small style="color:#475569">YOU</small><br><br>{query}</div>', unsafe_allow_html=True)

        # Progress area
        progress = st.empty()
        answer_area = st.empty()

        stages = []
        full_answer = ""

        try:
            for item in run_research_turn(st.session_state.session_id, query):
                if isinstance(item, AgentStep):
                    stages.append(item.message)
                    # Show all stages so far
                    progress.markdown(
                        "**Research in progress...**\n\n" +
                        "\n\n".join(f"• {s}" for s in stages)
                    )
                elif isinstance(item, str):
                    full_answer += item
                    answer_area.markdown(
                        f'<div class="assistant-msg">{full_answer}▌</div>',
                        unsafe_allow_html=True
                    )

            # Final clean render
            progress.empty()
            answer_area.empty()

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.code(traceback.format_exc())

        st.rerun()