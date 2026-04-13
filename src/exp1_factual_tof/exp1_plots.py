import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

def categorize_family(model_id):
    """Helper function to assign models to families based on their name."""
    name = str(model_id).lower()
    if 'deepseek' in name:
        return 'DeepSeek'
    elif 'qwen' in name:
        return 'Qwen'
    elif 'llama' in name or 'hermes' in name:
        return 'Llama'
    else:
        return 'Other'


def extract_model_size(model_id):
    """Extract numeric model size (e.g., 1.5, 7, 72) for Qwen ordering."""
    name = str(model_id).lower()
    match = re.search(r'(\d+(?:\.\d+)?)\s*b', name)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+(?:\.\d+)?)', name)
    return float(match.group(1)) if match else float('inf')


def get_family_model_order(family_df, family):
    """Return model order for each family based on experiment requirements."""
    unique_models = family_df['model_id'].drop_duplicates().tolist()

    if family == 'DeepSeek':
        # Ensure DeepSeek V3 variants appear before R1 variants.
        return sorted(
            unique_models,
            key=lambda m: (
                0 if 'v3' in str(m).lower() else 1 if 'r1' in str(m).lower() else 2,
                str(m).lower()
            )
        )

    if family == 'Qwen':
        # Sort by model size: 1.5, 7, 72, ...
        return sorted(unique_models, key=lambda m: (extract_model_size(m), str(m).lower()))

    if family == 'Llama':
        # Place base Llama variants before Hermes variants.
        return sorted(
            unique_models,
            key=lambda m: (
                0 if ('llama' in str(m).lower() and 'hermes' not in str(m).lower()) else 1 if 'hermes' in str(m).lower() else 2,
                str(m).lower()
            )
        )

    return sorted(unique_models, key=lambda m: str(m).lower())

def generate_sycophancy_plots(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Apply family categorization
    df['family'] = df['model_id'].apply(categorize_family)

    # Enforce language ordering across all plots.
    language_order = ['en', 'es', 'ca']
    df['language'] = pd.Categorical(df['language'], categories=language_order, ordered=True)
    
    # Set global aesthetic parameters for seaborn
    sns.set_theme(style="whitegrid")
    families = ['DeepSeek', 'Qwen', 'Llama']
    family_to_exp_type = {
        'DeepSeek': 'reasoning_vs_nonreasoning',
        'Llama': 'RLHF_vs_DPO',
        'Qwen': 'model_size'
    }
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Family-Specific Base Accuracy Plots
    # ---------------------------------------------------------
    for family in families:
        family_df = df[df['family'] == family]
        if family_df.empty:
            continue

        model_order = get_family_model_order(family_df, family)
        exp_type = family_to_exp_type[family]
            
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=family_df, 
            x='model_id', 
            y='base_accuracy', 
            hue='language', 
            palette='viridis',
            hue_order=language_order,
            order=model_order
        )
        plt.title(f'Base Accuracy - {exp_type}', fontsize=14, fontweight='bold')
        plt.ylabel('Base Accuracy (Proportion)')
        plt.xlabel('Model')
        plt.ylim(0, 1.05)
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"base_accuracy_{exp_type}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

    # ---------------------------------------------------------
    # 2. Family-Specific Turn of Flip (ToF) Plots
    # ---------------------------------------------------------
    for family in families:
        family_df = df[df['family'] == family]
        if family_df.empty:
            continue

        model_order = get_family_model_order(family_df, family)
        exp_type = family_to_exp_type[family]
            
        plt.figure(figsize=(10, 6))
        
        # Pivot for easy error bar plotting
        pivot_avg = family_df.pivot(index='model_id', columns='language', values='avg_tof')
        pivot_std = family_df.pivot(index='model_id', columns='language', values='std_tof')
        pivot_avg = pivot_avg.reindex(model_order)
        pivot_std = pivot_std.reindex(model_order)
        pivot_avg = pivot_avg.reindex(columns=language_order)
        pivot_std = pivot_std.reindex(columns=language_order)
        
        # Plot using pandas built-in matplotlib integration for yerr
        ax = pivot_avg.plot(
            kind='bar', 
            yerr=pivot_std, 
            capsize=4, 
            figsize=(10, 6),
            colormap='viridis',
            edgecolor='black'
        )
        
        plt.title(f'Average Turn of Flip - {exp_type}', fontsize=14, fontweight='bold')
        plt.ylabel('Average Turn of Flip (Higher = More Resistant)')
        plt.xlabel('Model')
        plt.xticks(rotation=0) # Keep labels horizontal if names aren't too long
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"tof_{exp_type}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

    # ---------------------------------------------------------
    # 3. Macro Scatter Plot: Base Accuracy vs. Average ToF
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='base_accuracy', 
        y='avg_tof', 
        hue='model_id', 
        style='language', 
        s=150, 
        palette='tab20'
    )
    plt.title('Base Accuracy vs. Average Turn of Flip (All Models)', fontsize=14, fontweight='bold')
    plt.xlabel('Base Accuracy')
    plt.ylabel('Average Turn of Flip')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    scatter_path = os.path.join(plots_dir, "scatter_accuracy_vs_tof.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {scatter_path}")

    # ---------------------------------------------------------
    # 4. Incorrect Proportion Plot (All Models)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x='model_id', 
        y='incorrect_proportion', 
        hue='language', 
        palette='magma',
        hue_order=language_order
    )
    plt.title('Format Break / Language Deviation Rate (All Models)', fontsize=14, fontweight='bold')
    plt.ylabel('Incorrect Proportion')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    incorrect_path = os.path.join(plots_dir, "incorrect_proportion_all.png")
    plt.savefig(incorrect_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {incorrect_path}")

    # ---------------------------------------------------------
    # 5. Aggregate Metrics DataFrame Preparation
    # ---------------------------------------------------------
    # Calculate the mean of metrics across all models for each language
    agg_df = df.groupby('language').agg({
        'base_accuracy': 'mean',
        'avg_tof': 'mean',
        'std_tof': 'mean', # Mean of standard deviations to approximate general variance
        'incorrect_proportion': 'mean'
    }).reset_index()
    agg_df = agg_df.set_index('language').reindex(language_order).dropna(how='all').reset_index()

    # ---------------------------------------------------------
    # 6. Aggregate Base Accuracy by Language
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=agg_df,
        x='language',
        y='base_accuracy',
        palette='viridis',
        edgecolor='black'
    )
    plt.title('Aggregate Base Accuracy by Language (All Models)', fontsize=14, fontweight='bold')
    plt.ylabel('Base Accuracy (Proportion)')
    plt.xlabel('Language')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    agg_base_path = os.path.join(plots_dir, "aggregate_base_accuracy_by_language.png")
    plt.savefig(agg_base_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {agg_base_path}")

    # ---------------------------------------------------------
    # 7. Aggregate Incorrect Proportion by Language
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=agg_df,
        x='language',
        y='incorrect_proportion',
        palette='magma',
        edgecolor='black'
    )
    plt.title('Aggregate Incorrect Proportion by Language (All Models)', fontsize=14, fontweight='bold')
    plt.ylabel('Incorrect Proportion')
    plt.xlabel('Language')
    plt.tight_layout()
    
    agg_incorrect_path = os.path.join(plots_dir, "aggregate_incorrect_proportion_by_language.png")
    plt.savefig(agg_incorrect_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {agg_incorrect_path}")

    # ---------------------------------------------------------
    # 8. Aggregate Turn of Flip by Language (with Error Bars)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    # We use matplotlib's standard bar plot here to easily apply custom pre-calculated error bars
    plt.bar(
        agg_df['language'],
        agg_df['avg_tof'],
        yerr=agg_df['std_tof'],
        capsize=5,
        color=sns.color_palette('viridis', n_colors=len(agg_df['language'])),
        edgecolor='black'
    )
    plt.title('Aggregate Average Turn of Flip by Language', fontsize=14, fontweight='bold')
    plt.ylabel('Average Turn of Flip')
    plt.xlabel('Language')
    plt.tight_layout()
    
    agg_tof_path = os.path.join(plots_dir, "aggregate_tof_by_language.png")
    plt.savefig(agg_tof_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {agg_tof_path}")

# To execute the function:
# generate_sycophancy_plots_separated('exp1.csv')

if __name__ == "__main__":
    # Example usage: generate_sycophancy_plots("sycophancy_experiment_results.csv")
    generate_sycophancy_plots("results/exp1.csv")