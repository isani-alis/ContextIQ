"""
ContextIQ — RAGAS Evaluation Suite

Runs 4 RAGAS metrics against the live retrieval + LLM pipeline:
  - Faithfulness       : answer is grounded in retrieved context
  - Answer Relevancy   : answer addresses the question
  - Context Precision  : retrieved chunks are relevant
  - Context Recall     : retrieval found the needed information

Usage (from project root):
    venv/bin/python eval/ragas_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

from backend.retrieval.vector_store import VectorStore
from backend.retrieval.reranker import rerank
from backend.agent.graph import AgentState, build_graph

# ------------------------------------------------------------------
# Eval dataset — question / ground_truth pairs
# Edit these to match the document(s) you have indexed
# ------------------------------------------------------------------
EVAL_SAMPLES = [
    {
        "question": "What is the candidate's name?",
        "ground_truth": "Isani Alis Aminbhai",
    },
    {
        "question": "What programming languages does the candidate know?",
        "ground_truth": "Java, Python, JavaScript, TypeScript, C, C++, SQL, Shell Script",
    },
    {
        "question": "Where does the candidate work?",
        "ground_truth": "SAP SuccessFactors",
    },
    {
        "question": "What is the candidate's CPI or GPA?",
        "ground_truth": "8.69 out of 10",
    },
    {
        "question": "What DevOps tools does the candidate use?",
        "ground_truth": "Docker, Kubernetes, Helm, Jenkins, GitHub Actions, CI/CD, Git, Linux",
    },
    {
        "question": "What is the candidate's Codeforces rating?",
        "ground_truth": "Expert rating of 1610",
    },
]


def run_pipeline(question: str, store: VectorStore, graph) -> tuple[str, list[str]]:
    """Run retrieval + LLM, return (answer, list_of_context_strings)."""
    initial_state: AgentState = {
        "query": question,
        "history": [],
        "candidates": [],
        "reranked": [],
        "confident": False,
        "answer": "",
        "citations": [],
    }
    result = graph.invoke(initial_state)

    # Collect retrieved context strings
    candidates = store.hybrid_search(question, top_k=15)
    reranked = rerank(question, candidates, top_k=5)
    contexts = [chunk.text for chunk, _ in reranked]

    return result["answer"], contexts


def main():
    print("Loading index...")
    if not VectorStore.exists():
        print("ERROR: No index found. Upload and index documents first via the UI.")
        sys.exit(1)

    store = VectorStore.load()

    print("Loading LLM (llama3.2 via Ollama)...")
    llm = ChatOllama(model="llama3.2", temperature=0)
    graph = build_graph(store, llm)

    print(f"Running {len(EVAL_SAMPLES)} eval samples...\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, sample in enumerate(EVAL_SAMPLES):
        print(f"[{i+1}/{len(EVAL_SAMPLES)}] {sample['question']}")
        answer, ctx = run_pipeline(sample["question"], store, graph)
        print(f"  → {answer[:120]}{'...' if len(answer) > 120 else ''}\n")

        questions.append(sample["question"])
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(sample["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Running RAGAS evaluation (this may take a minute)...")

    # Use Ollama for both LLM and embeddings in RAGAS
    ollama_llm = ChatOllama(model="llama3.2", temperature=0)
    ollama_emb = OllamaEmbeddings(model="llama3.2")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ollama_llm,
        embeddings=ollama_emb,
    )

    # Print results
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    scores = results.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metric_cols:
        if col in scores.columns:
            avg = scores[col].mean()
            print(f"  {col:<25} {avg:.4f}")

    print("="*50)

    # Save to file
    output_path = Path(__file__).parent / "results.json"
    result_dict = {
        col: float(scores[col].mean())
        for col in metric_cols if col in scores.columns
    }
    output_path.write_text(json.dumps(result_dict, indent=2))
    print(f"\nResults saved to eval/results.json")


if __name__ == "__main__":
    main()
