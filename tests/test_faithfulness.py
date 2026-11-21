"""
Faithfulness evaluation using DeepEval.

Usage:
    python tests/test_faithfulness.py
    python tests/test_faithfulness.py --threshold 0.6
    python tests/test_faithfulness.py --question "What chip does Mac Studio use?"

What it checks:
    FaithfulnessMetric scores whether every claim in the LLM answer
    is grounded in the retrieved chunks — no hallucination beyond the PDF.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq

# ── import RAG pipeline ──────────────────────────────────────────────────────
from chatbot import rag_query


# ── Groq judge model for DeepEval ────────────────────────────────────────────
class GroqJudge(DeepEvalBaseLLM):
    """Wraps Groq so DeepEval can use it as the faithfulness judge."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model_name = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name

# ── golden dataset ───────────────────────────────────────────────────────────
GOLDEN_DATASET = [
    {
        "id":       "TC001",
        "question": "What chip does Mac Studio use?",
        "category": "hardware",
    },
    {
        "id":       "TC002",
        "question": "How many CPU cores does M4 Max have?",
        "category": "hardware",
    },
    {
        "id":       "TC003",
        "question": "What storage options are available?",
        "category": "storage",
    },
    {
        "id":       "TC004",
        "question": "How many displays can M3 Ultra support?",
        "category": "hardware",
    },
    {
        "id":       "TC005",
        "question": "What is Apple Intelligence?",
        "category": "software",
    },
    {
        "id":       "TC006",
        "question": "How much faster is GPU performance?",
        "category": "performance",
    },
    {
        "id":       "TC007",
        "question": "What is the price of Mac Studio?",
        "category": "out-of-scope",
    },
    {
        "id":       "TC008",
        "question": "What is the design size of Mac Studio?",
        "category": "hardware",
    },
    {
        "id":       "TC009",
        "question": "Tell me about Mac Studio connectivity ports",
        "category": "hardware",
    },
    {
        "id":       "TC010",
        "question": "Does Mac Studio support iPhone?",
        "category": "out-of-scope",
    },
]


def run_evaluation(questions: list[dict], threshold: float) -> None:
    metric = FaithfulnessMetric(
        threshold=threshold,
        model=GroqJudge(),
        include_reason=True,
    )

    test_cases = []
    rag_results = {}

    print(f"\n{'─'*60}")
    print(f"  Running RAG pipeline for {len(questions)} question(s)...")
    print(f"{'─'*60}")

    for item in questions:
        qid = item["id"]
        question = item["question"]
        print(f"  [{qid}] {question}")

        result = rag_query(question)
        rag_results[qid] = result

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=result["answer"],
                retrieval_context=result["chunks"],
            )
        )

    print(f"\n{'─'*60}")
    print(f"  Scoring faithfulness (threshold={threshold})...")
    print(f"{'─'*60}\n")

    eval_results = evaluate(
        test_cases,
        [metric],
        display_config=DisplayConfig(print_results=False, inspect_after_run=False),
    )

    # ── print results ─────────────────────────────────────────────────────────
    passed = 0
    failed = 0
    scores = []

    print(f"{'ID':<8} {'Category':<14} {'Score':<8} {'Pass':<6}  Reason")
    print("─" * 90)

    for item, tc, test_result in zip(questions, test_cases, eval_results.test_results):
        metric_result = test_result.metrics_data[0]
        score  = round(metric_result.score, 3)
        status = "✅ PASS" if metric_result.success else "❌ FAIL"
        reason = metric_result.reason or ""

        scores.append(score)
        if metric_result.success:
            passed += 1
        else:
            failed += 1

        print(f"{item['id']:<8} {item['category']:<14} {score:<8} {status:<6}  {reason[:60]}")

        # show retrieved chunks summary
        r = rag_results[item["id"]]
        for i, (chunk, dist) in enumerate(zip(r["chunks"], r["distances"])):
            print(f"         chunk{i} dist={dist:.4f}  {chunk[:80].strip()}...")
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0
    print("─" * 90)
    print(f"  Total: {len(questions)}  ✅ {passed} passed  ❌ {failed} failed  avg_score={avg_score}")
    print(f"{'─'*60}\n")

    # ── save JSON report ──────────────────────────────────────────────────────
    report_path = ROOT / "logs" / "faithfulness_report.json"
    report = []
    for item, test_result in zip(questions, eval_results.test_results):
        md = test_result.metrics_data[0]
        r  = rag_results[item["id"]]
        report.append({
            "id":        item["id"],
            "category":  item["category"],
            "question":  item["question"],
            "answer":    r["answer"],
            "score":     round(md.score, 4),
            "passed":    md.success,
            "reason":    md.reason,
            "chunks":    [
                {"rank": i+1, "distance": round(r["distances"][i], 4), "content": r["chunks"][i]}
                for i in range(len(r["chunks"]))
            ],
        })

    with open(report_path, "w") as f:
        json.dump({"threshold": threshold, "results": report}, f, indent=2)
    print(f"  Report saved → {report_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepEval FaithfulnessMetric for Mac Studio RAG")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Faithfulness pass threshold (default: 0.5)")
    parser.add_argument("--question", type=str, default=None,
                        help="Run a single custom question instead of the golden dataset")
    args = parser.parse_args()

    if args.question:
        questions = [{"id": "CUSTOM", "question": args.question, "category": "custom"}]
    else:
        questions = GOLDEN_DATASET

    run_evaluation(questions, threshold=args.threshold)


if __name__ == "__main__":
    main()
