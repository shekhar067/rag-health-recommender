import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the Data ---
rag_df = pd.read_csv('outputs/rag_evaluation_results.csv')
baseline_df = pd.read_csv('outputs/baseline_evaluation_results.csv')

# --- 2. Calculate Average Scores ---
rag_means = rag_df[['bleu', 'rougeL', 'bertscore_f1']].mean()
baseline_means = baseline_df[['bleu', 'rougeL', 'bertscore_f1']].mean()

# --- 3. Prepare Data for Plotting ---
metrics = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
rag_scores = rag_means.values
baseline_scores = baseline_means.values

# --- 4. Create the Bar Chart ---
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, rag_scores, width, label='Personalized RAG', color='green', alpha=0.8)
rects2 = ax.bar(x + width/2, baseline_scores, width, label='Baseline LLM', color='blue', alpha=0.6)

# Add labels, title, and legend
ax.set_ylabel('Average Score', fontsize=12)
ax.set_title('Performance Comparison: Personalized RAG vs. Baseline LLM', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(rag_scores), max(baseline_scores)) * 1.2) # Set y-axis limit

# Add score labels on top of each bar
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

fig.tight_layout()
plt.show()
