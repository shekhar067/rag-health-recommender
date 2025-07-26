"""
Batch Evaluation & Visualization for RAG Health Recommender
- Accuracy, Faithfulness, Clarity (manual/auto), Novelty, Improvements after Fine-tuning
- Plots results for easy research reporting
"""

import json
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import os

# Import your pipeline
from rag_pipeline import rag_health_recommend

# --------- Configurable Evaluation Data (extend as needed) ---------
# Example: Provide sample questions and their gold-standard answers
SAMPLES = [
    {
        "query": "How should high blood pressure be treated?",
        "gold": "Lifestyle modification and medications such as ACE inhibitors."
    },
    {
        "query": "What is first-line therapy for type 2 diabetes?",
        "gold": "Metformin is first-line for type 2 diabetes."
    }
]

def exact_match(pred: str, gold: str) -> bool:
    # Simple, strict match (can use fuzzy/rouge/bert later)
    return gold.lower() in pred.lower()

def evaluate(samples: List[Dict], save_results: bool = True) -> List[Dict]:
    results = []
    for s in samples:
        out = rag_health_recommend(s["query"])
        result = {
            "query": s["query"],
            "gold": s["gold"],
            "prediction": out,
            "exact_match": exact_match(out, s["gold"])
        }
        results.append(result)
        print(f"Q: {s['query']}\nGOLD: {s['gold']}\nPRED: {out}\n")
    if save_results:
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
    return results

def visualize_accuracy(results: List[Dict]):
    scores = [int(r["exact_match"]) for r in results]
    acc = np.mean(scores)
    plt.figure(figsize=(4,4))
    plt.bar(["Accuracy"], [acc], color='green')
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("RAG Pipeline Accuracy")
    plt.show()

if __name__ == "__main__":
    results = evaluate(SAMPLES)
    visualize_accuracy(results)
    # Manual: add clarity/faithfulness/novelty columns in results.json for labeller

    print("\n--- For further research: ---")
    print("You can extend to:")
    print("- Use BLEU/ROUGE/BERTScore with NLG libraries")
    print("- Compare RAG vs non-RAG (plain FLAN-T5) by changing generator")
    print("- Add loss/novelty metrics from fine-tuning logs")
