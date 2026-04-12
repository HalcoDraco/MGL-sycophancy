import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_custom_model_order(models):
    """Helper function to sort models based on the requested sequence."""
    ordered_models = []
    
    # 1. Llama3.1 (Excluding Hermes)
    ordered_models.extend([m for m in models if 'llama' in str(m).lower() and 'hermes' not in str(m).lower() and m not in ordered_models])
    # 2. Llama Hermes version
    ordered_models.extend([m for m in models if 'hermes' in str(m).lower() and m not in ordered_models])
    # 3. Qwen 1.5B
    ordered_models.extend([m for m in models if '1.5b' in str(m).lower() and m not in ordered_models])
    # 4. Qwen 7B (Ensure we don't accidentally grab 72B)
    ordered_models.extend([m for m in models if '7b' in str(m).lower() and '72b' not in str(m).lower() and m not in ordered_models])
    # 5. Qwen 72B
    ordered_models.extend([m for m in models if '72b' in str(m).lower() and m not in ordered_models])
    
    # Append any remaining models just in case
    ordered_models.extend([m for m in models if m not in ordered_models])
    
    return ordered_models

def generate_system_prompt_plots(exp1_csv_path, exp2_csv_path):
    # Ensure the plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Load the datasets
    df1 = pd.read_csv(exp1_csv_path)
    df2 = pd.read_csv(exp2_csv_path)
    
    # Set global aesthetic parameters
    sns.set_theme(style="whitegrid")
    
    # Define exact categorical orderings
    all_models = pd.concat([df1['model_id'], df2['model_id']]).unique()
    model_order = get_custom_model_order(all_models)
    prompt_order = ['default', 'andrew', 'non-sycophantic', 'andrew + non-sycophantic']
    
    # Define a helper function for standard grouped bar plots with ordering
    def plot_grouped_bar(df, x_col, y_col, hue_col, title, ylabel, filename, ylim=None):
        plt.figure(figsize=(12, 6))
        
        # Filter order lists to include only items actually present in the data subset
        current_models = [m for m in model_order if m in df[x_col].unique()]
        current_prompts = [p for p in prompt_order if p in df[hue_col].unique()]

        sns.barplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            hue=hue_col,
            order=current_models,
            hue_order=current_prompts,
            palette='Set2',
            edgecolor='black'
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel)
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        if ylim:
            plt.ylim(ylim)
        plt.legend(title='System Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: plots/{filename}")

    # =========================================================
    # EXPERIMENT 1 PLOTS (False Presuppositions)
    # =========================================================
    
    # 1. Base Accuracy
    plot_grouped_bar(
        df=df1, x_col='model_id', y_col='base_accuracy', hue_col='system_prompt_key',
        title='Base Accuracy by Model and System Prompt',
        ylabel='Base Accuracy (Proportion)',
        filename='exp3_exp1_base_accuracy.png',
        ylim=(0, 1.05)
    )
    
    # 2. Incorrect Proportion
    plot_grouped_bar(
        df=df1, x_col='model_id', y_col='incorrect_proportion', hue_col='system_prompt_key',
        title='Format Break / Deviation Rate by Model and System Prompt',
        ylabel='Incorrect Proportion',
        filename='exp3_exp1_incorrect_proportion.png'
    )
    
    # 3. Average Turn of Flip (with Error Bars)
    # Using pandas bar plot for this since we want custom error bars via yerr
    pivot_avg = df1.pivot(index='model_id', columns='system_prompt_key', values='avg_tof')
    pivot_std = df1.pivot(index='model_id', columns='system_prompt_key', values='std_tof')
    
    # Apply our custom orderings directly to the pivot tables
    current_models_tof = [m for m in model_order if m in pivot_avg.index]
    current_prompts_tof = [p for p in prompt_order if p in pivot_avg.columns]
    
    pivot_avg = pivot_avg.reindex(index=current_models_tof, columns=current_prompts_tof)
    pivot_std = pivot_std.reindex(index=current_models_tof, columns=current_prompts_tof)
    
    ax = pivot_avg.plot(
        kind='bar', 
        yerr=pivot_std, 
        capsize=4, 
        figsize=(12, 6),
        colormap='Set2',
        edgecolor='black'
    )
    plt.title('Average Turn of Flip by Model and System Prompt', fontsize=14, fontweight='bold')
    plt.ylabel('Turn of Flip (Higher = More Resistant)')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='System Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/exp3_exp1_avg_tof.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plots/exp3_exp1_avg_tof.png")

    # =========================================================
    # EXPERIMENT 2 PLOTS (Moral Conflicts / AITA)
    # =========================================================
    
    # 4. All Yes Rate (Sycophancy)
    plot_grouped_bar(
        df=df2, x_col='model_id', y_col='all_yes_rate', hue_col='system_prompt_key',
        title='Always Agrees Rate (Sycophancy) by Model and System Prompt',
        ylabel='All Yes Rate (Proportion)',
        filename='exp3_exp2_all_yes_rate.png',
        ylim=(0, 1.05)
    )
    
    # 5. All No Rate (Contrarianism)
    plot_grouped_bar(
        df=df2, x_col='model_id', y_col='all_no_rate', hue_col='system_prompt_key',
        title='Always Disagrees Rate (Contrarianism) by Model and System Prompt',
        ylabel='All No Rate (Proportion)',
        filename='exp3_exp2_all_no_rate.png',
        ylim=(0, 1.05)
    )
    
    # 6. Reddit Agreement Rate (Human Alignment)
    plot_grouped_bar(
        df=df2, x_col='model_id', y_col='reddit_agree_rate', hue_col='system_prompt_key',
        title='Reddit Agreement Rate by Model and System Prompt',
        ylabel='Agreement Rate with Reddit Verdicts',
        filename='exp3_exp2_reddit_agreement.png',
        ylim=(0, 1.05)
    )

if __name__ == "__main__":
    exp1_csv = "results/exp1_res/exp1.csv"
    exp2_csv = "results/exp2_res/exp2.csv"
    generate_system_prompt_plots(exp1_csv, exp2_csv)

