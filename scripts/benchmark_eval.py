"""NeuraPilot Offline Evaluation Benchmark.

Evaluates the RAG pipeline on a test set of question-answer pairs and
computes RAGAS-style metrics (faithfulness, relevance, context precision).

Usage:
    python scripts/benchmark_eval.py --course ml101 --questions eval_qs.json

eval_qs.json format:
[
    {"question": "What is backpropagation?", "expected_answer": "..."},
    ...
]

Output: CSV + summary printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurapilot.config import get_settings
from neurapilot.core import db as dbmod
from neurapilot.evaluation.metrics import evaluate_response
from neurapilot.rag.agent_graph import build_pipeline
from neurapilot.rag.llm import build_llm_bundle
from neurapilot.rag.store import get_retriever


def run_benchmark(course_id: str, questions: list[dict], strict: bool = True) -> None:
    settings = get_settings()
    bundle = build_llm_bundle(settings)
    retriever = get_retriever(settings, bundle.embeddings, course_id)
    pipeline = build_pipeline(
        llm=bundle.llm,
        retriever=retriever,
        strict_default=strict,
        top_k=settings.top_k,
        hallucination_guard=True,
    )

    results = []
    for i, item in enumerate(questions, start=1):
        q = item["question"]
        print(f"[{i}/{len(questions)}] {q[:60]}...")

        t0 = time.time()
        state = pipeline.invoke({"question": q, "strict": strict})
        latency_ms = int((time.time() - t0) * 1000)

        output = state.get("output", "")
        docs = state.get("docs", []) or []

        scores = evaluate_response(bundle.llm, q, output, docs)

        result = {
            "question": q,
            "intent": state.get("intent", ""),
            "topic": state.get("topic", ""),
            "output": output[:200],
            "latency_ms": latency_ms,
            "faithfulness": scores.faithfulness,
            "answer_relevance": scores.answer_relevance,
            "context_precision": scores.context_precision,
            "mean_score": scores.mean_score(),
        }
        results.append(result)
        print(
            f"  → faithfulness={scores.faithfulness}, "
            f"relevance={scores.answer_relevance}, "
            f"precision={scores.context_precision}, "
            f"latency={latency_ms}ms"
        )

    # Summary
    valid_results = [r for r in results if r["mean_score"] is not None]
    if valid_results:
        avg_faith = sum(r["faithfulness"] or 0 for r in valid_results) / len(valid_results)
        avg_rel = sum(r["answer_relevance"] or 0 for r in valid_results) / len(valid_results)
        avg_prec = sum(r["context_precision"] or 0 for r in valid_results) / len(valid_results)
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        avg_mean = sum(r["mean_score"] or 0 for r in valid_results) / len(valid_results)

        print("\n" + "="*60)
        print(f"BENCHMARK RESULTS — Course: {course_id} — Mode: {'strict' if strict else 'tutor'}")
        print(f"{'Questions:':<25} {len(questions)}")
        print(f"{'Avg Faithfulness:':<25} {avg_faith:.3f}")
        print(f"{'Avg Answer Relevance:':<25} {avg_rel:.3f}")
        print(f"{'Avg Context Precision:':<25} {avg_prec:.3f}")
        print(f"{'Avg Mean Score:':<25} {avg_mean:.3f}")
        print(f"{'Avg Latency (ms):':<25} {avg_lat:.0f}")
        print("="*60)

    # Export CSV
    out_path = Path(f"eval_results_{course_id}_{int(time.time())}.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuraPilot Offline Evaluation")
    parser.add_argument("--course", required=True, help="Course ID to evaluate")
    parser.add_argument("--questions", required=True, help="Path to JSON file with test questions")
    parser.add_argument("--strict", action="store_true", default=True, help="Run in strict mode")
    parser.add_argument("--tutor", action="store_true", help="Run in tutor mode (overrides --strict)")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)

    strict = not args.tutor
    run_benchmark(args.course, questions, strict=strict)


if __name__ == "__main__":
    main()
