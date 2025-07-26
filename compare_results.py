import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Define your systems and locate the result files ---
output_dir = 'outputs'
# Add the baseline system to your configurations
systems_to_compare = {
    "Baseline LLM": os.path.join(output_dir, 'baseline_evaluation_results.csv'),
    "BioBERT + FLAN-T5": os.path.join(output_dir, 'rag_evaluation_biobert_flan-t5.csv'),
    "PubMedBERT + FLAN-T5": os.path.join(output_dir, 'rag_evaluation_pubmedbert_flan-t5.csv')
}

results_data = {}
for system_name, file_path in systems_to_compare.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        results_data[system_name] = df[['bleu', 'rougeL', 'bertscore_f1']].mean()
    else:
        print(f"Warning: Result file not found for '{system_name}'. Skipping. Path: {file_path}")

# Check if we have any data to process
if not results_data:
    print("No result files found. Please run the evaluation scripts first.")
else:
    # --- 2. Create and print a summary DataFrame ---
    summary_df = pd.DataFrame(results_data).T
    print("--- Final Performance Summary ---")
    print(summary_df.to_string(formatters={'bleu':'{:.3f}'.format, 'rougeL':'{:.3f}'.format, 'bertscore_f1':'{:.3f}'.format}))

    # --- 3. Create the final 3-way comparison chart ---
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    system_labels = list(summary_df.index)
    x = np.arange(len(metrics))
    width = 0.25 # Adjust width for three bars
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create a bar for each system
    for i, system_name in enumerate(system_labels):
        scores = summary_df.loc[system_name].values
        offset = width * (i - 1) # Center the bars
        rects = ax.bar(x + offset, scores, width, label=system_name)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=10)

    # Add labels, title, and legend
    ax.set_ylabel('Average Score', fontsize=14)
    ax.set_title('Final Comparison: Baseline vs. RAG Systems', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(title='System Configuration', fontsize=11)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_3-way_comparison_chart.png'))
    print("\nFinal 3-way comparison chart saved to 'outputs/final_3-way_comparison_chart.png'")
    plt.show()
