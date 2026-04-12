import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def generate_sycophancy_plots_separated(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Apply family categorization
    df['family'] = df['model_id'].apply(categorize_family)
    
    # Set global aesthetic parameters for seaborn
    sns.set_theme(style="whitegrid")
    families = ['DeepSeek', 'Qwen', 'Llama']
    
    # ---------------------------------------------------------
    # 1. Family-Specific Base Accuracy Plots
    # ---------------------------------------------------------
    for family in families:
        family_df = df[df['family'] == family]
        if family_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=family_df, 
            x='model_id', 
            y='base_accuracy', 
            hue='language', 
            palette='viridis'
        )
        plt.title(f'Base Accuracy - {family} Family', fontsize=14, fontweight='bold')
        plt.ylabel('Base Accuracy (Proportion)')
        plt.xlabel('Model')
        plt.ylim(0, 1.05)
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"base_accuracy_{family.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # ---------------------------------------------------------
    # 2. Family-Specific Turn of Flip (ToF) Plots
    # ---------------------------------------------------------
    for family in families:
        family_df = df[df['family'] == family]
        if family_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Pivot for easy error bar plotting
        pivot_avg = family_df.pivot(index='model_id', columns='language', values='avg_tof')
        pivot_std = family_df.pivot(index='model_id', columns='language', values='std_tof')
        
        # Plot using pandas built-in matplotlib integration for yerr
        ax = pivot_avg.plot(
            kind='bar', 
            yerr=pivot_std, 
            capsize=4, 
            figsize=(10, 6),
            colormap='viridis',
            edgecolor='black'
        )
        
        plt.title(f'Average Turn of Flip - {family} Family', fontsize=14, fontweight='bold')
        plt.ylabel('Average Turn of Flip (Higher = More Resistant)')
        plt.xlabel('Model')
        plt.xticks(rotation=0) # Keep labels horizontal if names aren't too long
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"tof_{family.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

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
    
    plt.savefig("scatter_accuracy_vs_tof.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: scatter_accuracy_vs_tof.png")

    # ---------------------------------------------------------
    # 4. Incorrect Proportion Plot (All Models)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x='model_id', 
        y='incorrect_proportion', 
        hue='language', 
        palette='magma'
    )
    plt.title('Format Break / Language Deviation Rate (All Models)', fontsize=14, fontweight='bold')
    plt.ylabel('Incorrect Proportion')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("incorrect_proportion_all.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: incorrect_proportion_all.png")

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
    
    plt.savefig("aggregate_base_accuracy_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: aggregate_base_accuracy_by_language.png")

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
    
    plt.savefig("aggregate_incorrect_proportion_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: aggregate_incorrect_proportion_by_language.png")

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
    
    plt.savefig("aggregate_tof_by_language.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: aggregate_tof_by_language.png")

# To execute the function:
# generate_sycophancy_plots_separated('exp1.csv')

if __name__ == "__main__":
    # Example usage: generate_sycophancy_plots("sycophancy_experiment_results.csv")
    generate_sycophancy_plots_separated("results/exp1.csv")