import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from rag_pipeline import rag_health_recommend

# -- If you want to compare, define the plain LLM (no retrieval) baseline --
def plain_llm_generate(user_query: str):
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt", device=-1, max_length=128)
    prompt = f"User question: {user_query}\nGive a clear, safe, evidence-based health recommendation."
    return generator(prompt)[0]['generated_text']

# ---- Your sample set ----
SAMPLES = [
    {
        "query": "How should high blood pressure be treated?",
        "gold": "Lifestyle modification and medications such as ACE inhibitors."
    },
    {
        "query": "What is first-line therapy for type 2 diabetes?",
        "gold": "Metformin remains a first-line drug for type 2 diabetes."
    }
    # Add more...
]

def exact_match(pred: str, gold: str) -> bool:
    return gold.lower() in pred.lower()

def compute_bleu(pred, gold):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)

def compute_rouge(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)['rougeL'].fmeasure

def evaluate(samples: List[Dict], pipeline_func, system_name="RAG"):
    results = []
    preds = []
    golds = []
    print(f"\n----- {system_name} Results -----")
    for s in samples:
        pred = pipeline_func(s["query"])
        preds.append(pred)
        golds.append(s["gold"])
        bleu = compute_bleu(pred, s["gold"])
        rouge = compute_rouge(pred, s["gold"])
        match = exact_match(pred, s["gold"])
        results.append({
            "system": system_name,
            "query": s["query"],
            "gold": s["gold"],
            "prediction": pred,
            "exact_match": match,
            "bleu": bleu,
            "rougeL": rouge,
        })
        print(f"\nQ: {s['query']}")
        print(f"GOLD: {s['gold']}")
        print(f"PRED: {pred}")
        print(f"BLEU: {bleu:.2f} | ROUGE-L: {rouge:.2f} | Exact Match: {match}")
    # BERTScore for all
    P, R, F1 = bert_score(preds, golds, lang="en", verbose=True)
    print(f"{system_name} BERTScore-F1 avg: {F1.mean():.3f}")
    for idx, r in enumerate(results):
        r['bertscore_f1'] = float(F1[idx])
    return results

def plot_metric(rag_results, base_results, metric, title, color="green", color2="gray"):
    rag_val = np.mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in rag_results])
    base_val = np.mean([r[metric] if metric != "exact_match" else int(r[metric]) for r in base_results])
    plt.figure(figsize=(4,3))
    plt.bar(["RAG", "Plain LLM"], [rag_val, base_val], color=[color, color2])
    plt.title(f"{title}: RAG vs Plain LLM")
    plt.ylim(0,1)
    plt.ylabel(title)
    plt.show()
    print(f"{title}: RAG = {rag_val:.2f}, Plain LLM = {base_val:.2f}")

# --------- Run everything and visualize! ---------
rag_results = evaluate(SAMPLES, rag_health_recommend, system_name="RAG")
base_results = evaluate(SAMPLES, plain_llm_generate, system_name="PlainLLM")

plot_metric(rag_results, base_results, "exact_match", "Accuracy")
plot_metric(rag_results, base_results, "bleu", "BLEU")
plot_metric(rag_results, base_results, "rougeL", "ROUGE-L")
plot_metric(rag_results, base_results, "bertscore_f1", "BERTScore-F1")

print("\n🎯 Done! All metrics and comparisons are above. Add more samples to SAMPLES for deeper analysis.")
