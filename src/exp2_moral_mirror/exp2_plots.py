import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def get_model_group(model_id):
    """Map model id to its comparison group label."""
    name = str(model_id).lower()
    if 'deepseek' in name:
        return 'reasoning_vs_nonreasoning'
    if 'llama' in name or 'hermes' in name:
        return 'RLHF_vs_DPO'
    if 'qwen' in name:
        return 'model_size'
    return 'other'

def get_custom_order(models):
    """Helper function to sort models based on the exact user-requested sequence."""
    ordered_models = []
    
    # 1. Deepseek V3
    ordered_models.extend([m for m in models if 'v3' in str(m).lower() and m not in ordered_models])
    # 2. Deepseek R1
    ordered_models.extend([m for m in models if 'r1' in str(m).lower() and m not in ordered_models])
    # 3. Llama3.1 (Excluding Hermes)
    ordered_models.extend([m for m in models if 'llama' in str(m).lower() and 'hermes' not in str(m).lower() and m not in ordered_models])
    # 4. Llama Hermes version
    ordered_models.extend([m for m in models if 'hermes' in str(m).lower() and m not in ordered_models])
    # 5. Qwen 1.5B
    ordered_models.extend([m for m in models if '1.5b' in str(m).lower() and m not in ordered_models])
    # 6. Qwen 7B (Ensure we don't accidentally grab 72B)
    ordered_models.extend([m for m in models if '7b' in str(m).lower() and '72b' not in str(m).lower() and m not in ordered_models])
    # 7. Qwen 72B
    ordered_models.extend([m for m in models if '72b' in str(m).lower() and m not in ordered_models])
    
    # Append any remaining models that didn't match the criteria
    ordered_models.extend([m for m in models if m not in ordered_models])
    
    return ordered_models

def generate_aita_experiment_plots(csv_path):
    # Ensure the plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Calculate the differentiated stance rate (cases where it didn't blindly agree or disagree)
    df['differentiated_rate'] = 1.0 - (df['all_yes_rate'] + df['all_no_rate'])
    
    # Set global aesthetic parameters
    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # 1. Reddit Agreement Rate (Sorted Bar Chart)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Sort models by Reddit agreement rate for a cleaner plot
    df_sorted_reddit = df.sort_values('reddit_agree_rate', ascending=False)
    
    sns.barplot(
        data=df_sorted_reddit,
        x='model_id',
        y='reddit_agree_rate',
        palette='Blues_r',
        edgecolor='black'
    )
    plt.title('Reddit Agreement Rate by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Agreement Rate with Reddit Verdicts')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    plt.savefig("plots/reddit_agreement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/reddit_agreement.png")

    # ---------------------------------------------------------
    # 2. Overall Stance Distribution (100% Stacked Bar Chart)
    # ---------------------------------------------------------
    # Apply custom sorting order to the DataFrame
    model_order = get_custom_order(df['model_id'].unique())
    df['model_id_cat'] = pd.Categorical(df['model_id'], categories=model_order, ordered=True)
    df_sorted_stance = df.sort_values('model_id_cat')
    
    # Prepare data for stacked bar chart
    df_stack = df_sorted_stance.set_index('model_id')[['all_yes_rate', 'all_no_rate', 'differentiated_rate']]
    
    # Plotting
    ax = df_stack.plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6),
        colormap='Pastel1',
        edgecolor='black'
    )

    # Add stronger group framing to compare both globally and within groups.
    group_colors = {
        'reasoning_vs_nonreasoning': '#cfe3ff',
        'RLHF_vs_DPO': '#ffe3bf',
        'model_size': '#cfeecf',
        'other': '#e8e8e8'
    }
    group_labels = {
        'reasoning_vs_nonreasoning': 'Reasoning vs Non-Reasoning',
        'RLHF_vs_DPO': 'RLHF vs DPO',
        'model_size': 'Model Size',
        'other': 'Other'
    }
    groups_in_order = [get_model_group(m) for m in model_order]

    boundaries = []
    current_group = groups_in_order[0] if groups_in_order else None
    start_idx = 0
    for idx, group in enumerate(groups_in_order + [None]):
        if group != current_group:
            end_idx = idx - 1
            boundaries.append((current_group, start_idx, end_idx))
            current_group = group
            start_idx = idx

    for group, start, end in boundaries:
        color = group_colors.get(group, '#e8e8e8')
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.1, zorder=0)

    # Draw separators between adjacent groups.
    for _, _, end in boundaries[:-1]:
        ax.axvline(end + 0.5, color='black', linestyle='-', linewidth=2.0)
    
    plt.title('Stance Distribution per Model', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion of Responses')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    
    # Move legend inside the plot, top-left
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Always Agrees (All Yes)', 'Always Disagrees (All No)', 'Takes a Stance (Differentiated)']
    plt.legend(handles, new_labels, loc='upper right')
    
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig("plots/stance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/stance_distribution.png")

    # ---------------------------------------------------------
    # 3. Scatter Plot: Sycophancy vs Reddit Agreement
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, 
        x='all_yes_rate', 
        y='reddit_agree_rate', 
        hue='model_id', 
        s=200, 
        palette='tab10',
        edgecolor='black'
    )
    plt.title('Sycophancy vs. Human Alignment (Reddit Agreement)', fontsize=14, fontweight='bold')
    plt.xlabel('Sycophancy Rate (All Yes Rate)')
    plt.ylabel('Reddit Agreement Rate')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("plots/scatter_sycophancy_vs_reddit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/scatter_sycophancy_vs_reddit.png")

    # ---------------------------------------------------------
    # 4. Scatter Plot: Contrarianism vs Reddit Agreement
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, 
        x='all_no_rate', 
        y='reddit_agree_rate', 
        hue='model_id', 
        s=200, 
        palette='tab10',
        edgecolor='black'
    )
    plt.title('Contrarianism vs. Human Alignment (Reddit Agreement)', fontsize=14, fontweight='bold')
    plt.xlabel('Contrarianism Rate (All No Rate)')
    plt.ylabel('Reddit Agreement Rate')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() 

    plt.savefig("plots/scatter_contrarianism_vs_reddit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/scatter_contrarianism_vs_reddit.png")

if __name__ == "__main__":
    generate_aita_experiment_plots('results/exp2.csv')