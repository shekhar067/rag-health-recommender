
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# --- Configuration ---
OUTPUT_DIR = 'outputs'
SYSTEMS_TO_COMPARE = {
    "Baseline LLM": os.path.join(OUTPUT_DIR, 'baseline_evaluation_results.csv'),
    "BioBERT + FLAN-T5": os.path.join(OUTPUT_DIR, 'rag_evaluation_biobert_flan-t5.csv'),
    "PubMedBERT + FLAN-T5": os.path.join(OUTPUT_DIR, 'rag_evaluation_pubmedbert_flan-t5.csv')
}
METRICS_TO_PLOT = ['bleu', 'rougeL', 'meteor', 'bertscore_f1', 'faithfulness', 'novelty']

# --- 1. Load and Combine Data ---
def load_all_data():
    """Loads all available CSV files into a single DataFrame."""
    all_results = []
    for system_name, file_path in SYSTEMS_TO_COMPARE.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['system'] = system_name
            all_results.append(df)
        else:
            print(f"Warning: Result file not found, skipping: {file_path}")
    
    if not all_results:
        print("Error: No result files found. Please run the orchestrator first.")
        return None, None
        
    combined_df = pd.concat(all_results)
    # Ensure metric columns exist, fill with 0 if not
    for metric in METRICS_TO_PLOT:
        if metric not in combined_df.columns:
            combined_df[metric] = 0.0
            
    return combined_df, list(SYSTEMS_TO_COMPARE.keys())

# --- 2. Visualization Functions ---

def plot_overall_performance_bar_chart(summary_df):
    """Generates the main bar chart of average scores."""
    print("\n--- Generating Chart 1: Overall Performance Bar Chart ---")
    metrics_labels = [m.replace('_', '-').replace('bertscore-f1', 'BERTScore-F1').replace('rougeL', 'ROUGE-L').upper() for m in METRICS_TO_PLOT]
    system_labels = list(summary_df.index)
    x = np.arange(len(metrics_labels))
    width = 0.8 / len(system_labels)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, system_name in enumerate(system_labels):
        scores = summary_df.loc[system_name].values
        offset = width * (i - (len(system_labels) - 1) / 2)
        rects = ax.bar(x + offset, scores, width, label=system_name)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)

    ax.set_ylabel('Average Score', fontsize=14)
    ax.set_title('Overall Model Performance Comparison', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=12)
    ax.legend(title='System Configuration', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_1_overall_performance.png'))
    plt.show()

def plot_score_distribution_box_plot(combined_df):
    """Generates a box plot to show the distribution of scores."""
    print("\n--- Generating Chart 2: Score Distribution Box Plot ---")
    melted_df = combined_df.melt(id_vars=['system'], value_vars=METRICS_TO_PLOT, var_name='metric', value_name='score')
    # NEW LINE - Correct
    melted_df['metric'] = melted_df['metric'].str.replace('_', '-').str.replace('bertscore-f1', 'BERTScore-F1').str.replace('rougeL', 'ROUGE-L').str.upper()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=melted_df, x='metric', y='score', hue='system', ax=ax)

    ax.set_title('Distribution of Evaluation Scores Across Systems', fontsize=18, pad=20)
    ax.set_xlabel('Metric', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.legend(title='System Configuration')
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_2_score_distribution.png'))
    plt.show()

def plot_radar_chart(summary_df):
    """Generates a radar chart for a multi-dimensional view."""
    print("\n--- Generating Chart 3: Radar Chart ---")
    labels = np.array([m.upper() for m in METRICS_TO_PLOT])
    stats = summary_df.values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, stats[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, row in enumerate(stats):
        ax.plot(angles, row, label=summary_df.index[i])
        ax.fill(angles, row, alpha=0.1)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Multi-Metric Performance Comparison", size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_3_radar_chart.png'))
    plt.show()

def create_side_by_side_table(combined_df):
    """Creates a pivot table for direct answer comparison."""
    print("\n--- Generating Table 4: Detailed Side-by-Side Comparison ---")
    comparison_table = combined_df.pivot_table(
        index=['query', 'gold'],
        columns='system',
        values='prediction',
        aggfunc=lambda x: ' '.join(x)
    ).reset_index()
    
    cols = ['query', 'gold'] + [col for col in SYSTEMS_TO_COMPARE.keys() if col in comparison_table.columns]
    comparison_table = comparison_table[cols]
    
    pd.set_option('display.max_colwidth', None)
    print(comparison_table.to_markdown(index=False))
    comparison_table.to_csv(os.path.join(OUTPUT_DIR, 'table_4_side_by_side_comparison.csv'), index=False)

def create_performance_heatmap(summary_df):
    """Creates a heatmap of the summary scores."""
    print("\n--- Generating Chart 5: Performance Metrics Heatmap ---")
    plt.figure(figsize=(10, 6))
    sns.heatmap(summary_df, annot=True, cmap="viridis", fmt=".3f", linewidths=.5)
    plt.title("Performance Metrics Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_5_performance_heatmap.png'))
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    combined_df, system_names = load_all_data()
    
    if combined_df is not None:
        summary_df = combined_df.groupby('system')[METRICS_TO_PLOT].mean().reindex(system_names)
        
        plot_overall_performance_bar_chart(summary_df)
        plot_score_distribution_box_plot(combined_df)
        plot_radar_chart(summary_df)
        create_side_by_side_table(combined_df)
        create_performance_heatmap(summary_df)
        
        print(f"\n✅ Advanced analysis complete. All charts and tables saved in '{OUTPUT_DIR}' directory.")
