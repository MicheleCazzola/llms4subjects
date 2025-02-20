import pandas as pd
import os
import matplotlib.pyplot as plt

# Replace 'your_directory_path' with the path to your directory
directory_path = 'results'

# List all files in the directory and subdirectories
excel_files = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.xlsx'):
            excel_files.append(os.path.join(root, file))


# Filter files containing 'finetuned' in the name
finetuned_files = [file for file in excel_files if 'finetuned' in os.path.basename(file)]

# Print the filtered list of files
metrics = ['precision', 'recall', 'f1']
titles = ['Top-k Precision', 'Top-k Recall', 'Top-k F1']
y_labels = ['Precision (%)', 'Recall (%)', 'F1 (%)']

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title, y_label in zip(axs, metrics, titles, y_labels):
    for file in finetuned_files:
        split = os.path.basename(file).split('.')[0].split('_')
        model = split[-4]
        loss = split[-3]
        df = pd.read_excel(file)
        metric_columns = [col for col in df.columns if metric in col]
        selected = df[df.iloc[:, 0] == 'Overall'][metric_columns]
        ax.plot(selected.columns, selected.values[0] * 100, 'o', label=loss, linestyle='-.', linewidth=1)  # Dashed line between points with smaller marker size
        ax.set_xticks(range(len(selected.columns)))
        ax.set_xticklabels([col.split("_")[-1] for col in selected.columns])

    baseline = [file for file in excel_files if 'all-MiniLM-L6-v2' in os.path.basename(file) and 'finetuned' not in os.path.basename(file)][0]
    # Plot baseline
    df_baseline = pd.read_excel(baseline)
    metric_columns_baseline = [col for col in df_baseline.columns if metric in col]
    selected_baseline = df_baseline[df_baseline.iloc[:, 0] == 'Overall'][metric_columns_baseline]
    ax.plot(selected_baseline.columns, selected_baseline.values[0] * 100, 's', label='Baseline', linestyle='solid', linewidth=1)  # Plot singular points with square marker and dashed line

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('k')
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='solid', alpha=0.5)
    fig.suptitle(f'Performance Metrics for finetuned {model} with different losses', fontsize=16)
plt.tight_layout()
plt.savefig('images/finetuned_metrics.svg')
plt.savefig('images/finetuned_metrics.jpg')
#plt.show()

mlp_files = [file for file in excel_files if 'mlp' in os.path.basename(file)]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title, y_label in zip(axs, metrics, titles, y_labels):
    for file in mlp_files:
        model = os.path.basename(file).split('.')[0].split('_')[1]
        df = pd.read_excel(file)
        metric_columns = [col for col in df.columns if metric in col]
        selected = df[df.iloc[:, 0] == 'Overall'][metric_columns]
        ax.plot(selected.columns, selected.values[0] * 100, 'o', label=f'{model} with MLP', linestyle='-.', linewidth=1)  # Dashed line between points with smaller marker size
        ax.set_xticks(range(len(selected.columns)))
        ax.set_xticklabels([col.split("_")[-1] for col in selected.columns])

        baseline = [file for file in excel_files if model in os.path.basename(file) and 
                'mlp' not in os.path.basename(file) and 'finetuned' not in os.path.basename(file)][0]
        # Plot baseline
        df_baseline = pd.read_excel(baseline)
        metric_columns_baseline = [col for col in df_baseline.columns if metric in col]
        selected_baseline = df_baseline[df_baseline.iloc[:, 0] == 'Overall'][metric_columns_baseline]
        ax.plot(selected_baseline.columns, selected_baseline.values[0] * 100, 's', label=f'{model} with CosSim', linestyle='solid', linewidth=1)  # Plot singular points with square marker and dashed line
    
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('k')
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='solid', alpha=0.5)
    fig.suptitle('Performance Metrics using MLP', fontsize=16)
plt.tight_layout()
plt.savefig('images/mlp_metrics.svg')
plt.savefig('images/mlp_metrics.jpg')
#plt.show()

other_files = [file for file in excel_files if file not in finetuned_files and file not in mlp_files]
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for ax, metric, title, y_label in zip(axs, metrics, titles, y_labels):
    for file in other_files:
        model = os.path.basename(file).split('.')[0].split('_')[0]
        df = pd.read_excel(file)
        metric_columns = [col for col in df.columns if metric in col]
        selected = df[df.iloc[:, 0] == 'Overall'][metric_columns]
        ax.plot(selected.columns, selected.values[0] * 100, 'o', label=f'{model}', linestyle='-.', linewidth=1)  # Dashed line between points with smaller marker size
        ax.set_xticks(range(len(selected.columns)))
        ax.set_xticklabels([col.split("_")[-1] for col in selected.columns])

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('k')
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='solid', alpha=0.5)
    fig.suptitle('Performance Metrics using Cosine Similarity', fontsize=16)
plt.tight_layout()
plt.savefig('images/cosine_similarity_metrics.svg')
plt.savefig('images/cosine_similarity_metrics.jpg')
#plt.show()