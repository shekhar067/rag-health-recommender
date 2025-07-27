import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Dict, List

# --- 1. Define your systems and locate the result files ---
output_dir = 'outputs'
systems_to_compare = {
    "Baseline LLM": os.path.join(output_dir, 'baseline_evaluation_results.csv'),
    "BioBERT + FLAN-T5": os.path.join(output_dir, 'rag_evaluation_biobert_flan-t5.csv'),
    "PubMedBERT + FLAN-T5": os.path.join(output_dir, 'rag_evaluation_pubmedbert_flan-t5.csv')
}

results_data: Dict[str, pd.Series] = {}
for system_name, file_path in systems_to_compare.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        results_data[system_name] = df[['bleu', 'rougeL', 'bertscore_f1', 'meteor', 'f1', 'em', 'clinical_accuracy', 'novelty', 'coverage', 'answer_length', 'faithfulness']].mean()
    else:
        print(f"Warning: Result file not found for '{system_name}'. Skipping. Path: {file_path}")

if not results_data:
    print("No result files found. Please run the evaluation scripts first.")
else:
    summary_df = pd.DataFrame(results_data).T
    print("--- Final Performance Summary ---")
    print(summary_df.to_string(formatters={
        'bleu': '{:.3f}'.format, 'rougeL': '{:.3f}'.format, 'bertscore_f1': '{:.3f}'.format,
        'meteor': '{:.3f}'.format, 'f1': '{:.3f}'.format, 'em': '{:.3f}'.format,
        'clinical_accuracy': '{:.3f}'.format, 'novelty': '{:.3f}'.format, 'coverage': '{:.3f}'.format,
        'answer_length': '{:.0f}'.format, 'faithfulness': '{:.3f}'.format
    }))

    # --- 3. Bar Chart: 3-way Comparison ---
    metrics_bar = ['BLEU', 'ROUGE-L', 'BERTScore-F1']
    system_labels = list(summary_df.index)
    x = np.arange(len(metrics_bar))
    width = 0.25

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    for i, system_name in enumerate(system_labels):
        scores = [summary_df.loc[system_name, 'bleu'], summary_df.loc[system_name, 'rougeL'], summary_df.loc[system_name, 'bertscore_f1']]
        offset = width * (i - 1)
        rects = ax1.bar(x + offset, scores, width, label=system_name)
        ax1.bar_label(rects, padding=3, fmt='%.2f', fontsize=10)

    ax1.set_ylabel('Average Score', fontsize=14)
    ax1.set_title('Final 3-way Comparison: Baseline vs. RAG Systems', fontsize=18, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_bar, fontsize=14)
    ax1.legend(title='System Configuration', fontsize=11)
    ax1.set_ylim(0, 1)
    fig1.tight_layout()
    plt1_path = os.path.join(output_dir, 'final_3-way_comparison_chart.png')
    fig1.savefig(plt1_path)
    print(f"\nFinal 3-way comparison chart saved to '{plt1_path}'")
    plt.close(fig1)

    # --- 4. Line Chart: Trend Across Queries ---
    sample_system = "PubMedBERT + FLAN-T5"
    if os.path.exists(systems_to_compare[sample_system]):
        query_df = pd.read_csv(systems_to_compare[sample_system])
        queries = [f"Query {i+1}" for i in range(len(query_df))]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for metric in ['rougeL', 'coverage']:
            ax2.plot(queries, query_df[metric], marker='o', label=f"{metric.upper()} ({sample_system})")
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Metric Trend Across Queries', fontsize=16)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        plt2_path = os.path.join(output_dir, 'query_trend_chart.png')
        fig2.savefig(plt2_path)
        print(f"Query trend chart saved to '{plt2_path}'")
        plt.close(fig2)

    # --- 5. Radar Chart: Multi-Metric Comparison ---
    metrics_radar = ['bleu', 'rougeL', 'meteor', 'coverage']
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = plt.subplot(111, polar=True)
    angles = [n / float(len(metrics_radar)) * 2 * np.pi for n in range(len(metrics_radar))]
    angles += angles[:1]
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    plt.xticks(angles[:-1], [m.upper() for m in metrics_radar], fontsize=12)
    ax3.set_rlabel_position(0)
    ax3.set_ylim(0, 1)
    for system_name in system_labels:
        values = [summary_df.loc[system_name, m] for m in metrics_radar]
        values += values[:1]
        ax3.plot(angles, values, linewidth=2, linestyle='solid', label=system_name)
        ax3.fill(angles, values, alpha=0.1)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    ax3.set_title('Multi-Metric Comparison', fontsize=16, pad=20)
    plt3_path = os.path.join(output_dir, 'radar_comparison_chart.png')
    fig3.savefig(plt3_path)
    print(f"Radar comparison chart saved to '{plt3_path}'")
    plt.close(fig3)

    # --- 6. Pie Chart: Clinical Accuracy Distribution ---
    if os.path.exists(systems_to_compare[sample_system]):
        query_df = pd.read_csv(systems_to_compare[sample_system])
        acc_counts = [len(query_df[query_df['clinical_accuracy'] == 1]), len(query_df[query_df['clinical_accuracy'] == 0])]
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.pie(acc_counts, labels=['Accurate', 'Inaccurate'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
        ax4.axis('equal')
        ax4.set_title('Clinical Accuracy Distribution', fontsize=14)
        plt4_path = os.path.join(output_dir, 'clinical_accuracy_pie_chart.png')
        fig4.savefig(plt4_path)
        print(f"Clinical accuracy pie chart saved to '{plt4_path}'")
        plt.close(fig4)

    # --- 7. Per-Question Heatmap ---
    if os.path.exists(systems_to_compare[sample_system]):
        query_df = pd.read_csv(systems_to_compare[sample_system])
        metrics_heatmap = ['bleu', 'rougeL', 'coverage', 'novelty']
        heatmap_data = query_df[metrics_heatmap].T
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax5)
        ax5.set_title('Per-Question Heatmap', fontsize=16)
        ax5.set_xlabel('Query Index')
        ax5.set_ylabel('Metric')
        plt5_path = os.path.join(output_dir, 'per_question_heatmap.png')
        fig5.savefig(plt5_path)
        print(f"Per-question heatmap saved to '{plt5_path}'")
        plt.close(fig5)

    # --- 8. Scatter Plot: Novelty vs. Coverage ---
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    for system_name in system_labels:
        if os.path.exists(systems_to_compare[system_name]):
            query_df = pd.read_csv(systems_to_compare[system_name])
            ax6.scatter(query_df['novelty'], query_df['coverage'], label=system_name, alpha=0.6)
    ax6.set_xlabel('Novelty', fontsize=12)
    ax6.set_ylabel('Coverage', fontsize=12)
    ax6.set_title('Novelty vs. Coverage', fontsize=16)
    ax6.legend(fontsize=10)
    ax6.grid(True)
    plt6_path = os.path.join(output_dir, 'novelty_vs_coverage_scatter.png')
    fig6.savefig(plt6_path)
    print(f"Novelty vs. Coverage scatter chart saved to '{plt6_path}'")
    plt.close(fig6)
