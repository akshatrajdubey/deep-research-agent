"""
Evaluation harness for the Deep Research Agent.

Metrics:
1. Citation Rate        — fraction of answers that contain at least one [N] citation
2. Source Grounding     — fraction of cited sources that were actually fetched (not hallucinated)
3. Conflict Handling    — does agent explicitly note conflicting sources when present?
4. Uncertainty Flagging — does agent hedge on low-evidence questions?
5. Answer Relevance     — keyword overlap between expected answer terms and actual answer

Run:
    python eval_harness.py
"""

import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import session as session_mod
from modules.agent import run_research_turn, AgentStep
from modules.session import init_db

# ─── Evaluation Dataset ───────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id": "factual_1",
        "type": "factual",
        "query": "What is the current interest rate set by the Federal Reserve?",
        "expected_keywords": ["federal reserve", "interest rate", "percent", "basis points", "fed"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Recent factual — should find current rate with citation"
    },
    {
        "id": "factual_2",
        "type": "factual",
        "query": "What programming language was Python named after?",
        "expected_keywords": ["monty python", "comedy", "guido", "van rossum"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Stable fact — should retrieve accurate info with citation"
    },
    {
        "id": "multihop_1",
        "type": "multi_hop",
        "query": "Who is the CEO of the company that makes the M1 chip, and what university did they attend?",
        "expected_keywords": ["apple", "tim cook", "auburn", "chip", "m1"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Multi-hop: company → CEO → education"
    },
    {
        "id": "comparison_1",
        "type": "comparison",
        "query": "What are the main differences between Python and JavaScript for backend development?",
        "expected_keywords": ["python", "javascript", "node", "django", "flask", "backend"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Comparison — should cover both sides with sources"
    },
    {
        "id": "low_evidence_1",
        "type": "insufficient_evidence",
        "query": "What will the exact stock price of Apple be on January 1st 2030?",
        "expected_keywords": [],
        "requires_citation": False,
        "expect_conflict": False,
        "expect_uncertainty": True,
        "description": "Unanswerable future prediction — should flag uncertainty"
    },
    {
        "id": "conflict_1",
        "type": "conflicting_sources",
        "query": "Is coffee good or bad for health?",
        "expected_keywords": ["coffee", "health", "studies", "research"],
        "requires_citation": True,
        "expect_conflict": True,
        "expect_uncertainty": False,
        "description": "Contested topic — sources likely conflict, agent should note disagreement"
    },
    {
        "id": "recent_1",
        "type": "recent_events",
        "query": "What are the latest developments in large language model research in 2025?",
        "expected_keywords": ["llm", "model", "ai", "language", "research", "2025"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Recent events — requires fresh web data"
    },
    {
        "id": "multiturn_1",
        "type": "multi_turn",
        "query": "Tell me about the history of the internet",
        "followup_query": "Who invented the World Wide Web specifically?",
        "expected_keywords": ["berners-lee", "www", "web", "cern"],
        "requires_citation": True,
        "expect_conflict": False,
        "expect_uncertainty": False,
        "description": "Multi-turn: context should carry over"
    },
]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def metric_citation_rate(answer: str) -> float:
    """Does the answer contain at least one [N] citation?"""
    return 1.0 if re.search(r'\[\d+\]', answer) else 0.0


def metric_source_grounding(answer: str, fetched_urls: list[str]) -> float:
    """
    Fraction of [N] citation numbers that correspond to actual fetched sources.
    Simple heuristic: if citations exist and sources were fetched, assume grounded.
    """
    citation_matches = re.findall(r'\[(\d+)\]', answer)
    if not citation_matches:
        return 0.0 if fetched_urls else 1.0  # no citations, no sources → N/A → treat as 1.0
    if not fetched_urls:
        return 0.0
    max_cited = max(int(n) for n in citation_matches)
    return min(1.0, len(fetched_urls) / max(max_cited, 1))


def metric_conflict_detection(answer: str, expects_conflict: bool) -> float:
    """Does agent note conflicting sources when expected?"""
    conflict_phrases = [
        "conflict", "contradict", "disagree", "differ", "however",
        "on the other hand", "while some sources", "sources disagree",
        "some studies", "mixed evidence"
    ]
    has_conflict_language = any(p in answer.lower() for p in conflict_phrases)
    if expects_conflict:
        return 1.0 if has_conflict_language else 0.0
    else:
        return 1.0  # Not expected — no penalty


def metric_uncertainty_flagging(answer: str, expects_uncertainty: bool) -> float:
    """Does agent appropriately flag uncertainty for unanswerable questions?"""
    uncertainty_phrases = [
        "cannot predict", "impossible to know", "uncertain", "unclear",
        "no evidence", "limited evidence", "speculative", "unpredictable",
        "not possible to determine", "future", "cannot be determined"
    ]
    has_uncertainty = any(p in answer.lower() for p in uncertainty_phrases)
    if expects_uncertainty:
        return 1.0 if has_uncertainty else 0.0
    else:
        return 1.0


def metric_answer_relevance(answer: str, expected_keywords: list[str]) -> float:
    """Keyword recall: fraction of expected terms found in answer."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def score_turn(case: dict, answer: str, fetched_urls: list[str]) -> dict:
    """Compute all metrics for one test case."""
    return {
        "citation_rate": metric_citation_rate(answer),
        "source_grounding": metric_source_grounding(answer, fetched_urls),
        "conflict_detection": metric_conflict_detection(answer, case.get("expect_conflict", False)),
        "uncertainty_flagging": metric_uncertainty_flagging(answer, case.get("expect_uncertainty", False)),
        "answer_relevance": metric_answer_relevance(answer, case.get("expected_keywords", [])),
    }


def composite_score(metrics: dict) -> float:
    weights = {
        "citation_rate": 0.25,
        "source_grounding": 0.25,
        "conflict_detection": 0.15,
        "uncertainty_flagging": 0.15,
        "answer_relevance": 0.20,
    }
    return sum(metrics.get(k, 0) * w for k, w in weights.items())


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_eval_case(case: dict, verbose: bool = True) -> dict:
    """Run a single evaluation case and return results."""
    init_db()
    sid = session_mod.create_session(f"eval_{case['id']}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[{case['type'].upper()}] {case['id']}: {case['description']}")
        print(f"Query: {case['query']}")
        print("-" * 60)

    answer = ""
    fetched_urls = []
    stages_seen = []
    start = time.time()

    # Run primary query
    for item in run_research_turn(sid, case["query"]):
        if isinstance(item, AgentStep):
            stages_seen.append(item.stage)
            if verbose:
                print(f"  [{item.stage}] {item.message}")
            if item.stage == "fetching" and "pages" in item.data:
                fetched_urls = [p["url"] for p in item.data["pages"] if not p.get("error")]
        elif isinstance(item, str):
            answer += item

    elapsed = time.time() - start

    # Optional multi-turn followup
    if "followup_query" in case:
        if verbose:
            print(f"\n  [FOLLOWUP] {case['followup_query']}")
        followup_answer = ""
        for item in run_research_turn(sid, case["followup_query"]):
            if isinstance(item, str):
                followup_answer += item
        # Use followup answer for evaluation
        answer = followup_answer

    metrics = score_turn(case, answer, fetched_urls)
    comp = composite_score(metrics)

    result = {
        "id": case["id"],
        "type": case["type"],
        "query": case["query"],
        "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
        "fetched_urls": fetched_urls,
        "stages_seen": stages_seen,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "composite_score": round(comp, 3),
    }

    if verbose:
        print(f"\n  Answer preview: {result['answer_preview'][:200]}...")
        print(f"  Metrics: {json.dumps(metrics, indent=2)}")
        print(f"  Composite: {comp:.3f}  |  Time: {elapsed:.1f}s")

    return result


def run_eval_suite(verbose: bool = True) -> dict:
    """Run all eval cases and produce a summary report."""
    results = []
    for case in EVAL_DATASET:
        try:
            r = run_eval_case(case, verbose=verbose)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {case['id']}: {e}")
            results.append({"id": case["id"], "error": str(e), "composite_score": 0.0})

    # Aggregate
    scores = [r.get("composite_score", 0) for r in results]
    avg = sum(scores) / len(scores) if scores else 0

    by_type = {}
    for r in results:
        t = r.get("type", "unknown")
        by_type.setdefault(t, []).append(r.get("composite_score", 0))

    type_avgs = {t: round(sum(v)/len(v), 3) for t, v in by_type.items()}

    metric_keys = ["citation_rate", "source_grounding", "conflict_detection",
                   "uncertainty_flagging", "answer_relevance"]
    metric_avgs = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in results if "metrics" in r]
        metric_avgs[k] = round(sum(vals)/len(vals), 3) if vals else 0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "overall_composite": round(avg, 3),
        "by_type": type_avgs,
        "metric_averages": metric_avgs,
        "cases": results,
    }

    # Save results
    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(results)}")
    print(f"Overall composite score: {avg:.3f}")
    print(f"\nBy type: {json.dumps(type_avgs, indent=2)}")
    print(f"\nMetric averages: {json.dumps(metric_avgs, indent=2)}")
    print(f"\nResults saved to: {out_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Deep Research Agent")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--case", type=str, help="Run single case by id")
    args = parser.parse_args()

    if args.case:
        matching = [c for c in EVAL_DATASET if c["id"] == args.case]
        if matching:
            run_eval_case(matching[0], verbose=not args.quiet)
        else:
            print(f"Case '{args.case}' not found. Available: {[c['id'] for c in EVAL_DATASET]}")
    else:
        run_eval_suite(verbose=not args.quiet)
