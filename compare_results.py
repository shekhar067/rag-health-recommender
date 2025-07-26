import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Define your configurations and locate the result files ---
output_dir = 'outputs'
configurations = [
    "biobert_flan-t5",
    "pubmedbert_flan-t5",
    "biobert_medalpaca",
    "pubmedbert_medalpaca"
]

results_data = {}
for config in configurations:
    file_path = os.path.join(output_dir, f'rag_evaluation_{config}.csv')
    try:
        df = pd.read_csv(file_path)
        # Calculate mean scores for this configuration
        results_data[config.replace('_', ' + ').title()] = df[['bleu', 'rougeL', 'bertscore_f1']].mean()
    except FileNotFoundError:
        print(f"Warning: Result file not found for configuration '{config}'. Skipping. Path: {file_path}")

# Check if we have any data to process
if not results_data:
    print("No result files found. Please run the evaluation scripts first.")
else:
    # --- 2. Create and print a summary DataFrame ---
    summary_df = pd.DataFrame(results_data).T # Transpose to have configs as rows
    print("--- Performance Summary Table ---")
    print(summary_df.to_string(formatters={'bleu':'{:.3f}'.format, 'rougeL':'{:.3f}'.format, 'bertscore_f1':'{:.3f}'.format}))


    # --- 3. Create the final comparison bar chart ---
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    configs_labels = list(summary_df.index)
    x = np.arange(len(metrics))  # the label locations
    width = 0.15  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create a bar for each configuration
    for i, config_name in enumerate(configs_labels):
        scores = summary_df.loc[config_name].values
        offset = width * (i - (len(configs_labels) - 1) / 2)
        rects = ax.bar(x + offset, scores, width, label=config_name)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    # Add labels, title, and legend
    ax.set_ylabel('Average Score', fontsize=14)
    ax.set_title('Overall Model Performance Comparison', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(title='Configurations', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_comparison_chart.png'))
    print("\nFinal comparison chart saved to 'outputs/final_comparison_chart.png'")
    plt.show()
