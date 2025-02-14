import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_results(file_path):
    """Load results from a jsonl file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_metrics(df, title):
    """Create a bar chart showing the proportion of unsafe responses with error bars."""
    plt.figure(figsize=(10, 6))
    
    df['total'] = df['unsafe'] + df['safe']
    df['unsafe_ratio'] = df['unsafe'] / df['total']
    
    df['model'] = df['redteam_model'].apply(lambda x: x.split('/')[-1].replace('-Instruct', ''))
    
    n = 500  # Number of trials
    df['std_error'] = (df['unsafe_ratio'] * (1 - df['unsafe_ratio']) / n) ** 0.5
    
    df_sorted = df.sort_values(by='model')
    
    bars = plt.bar(df_sorted['model'], df_sorted['unsafe_ratio'], yerr=df_sorted['std_error'], capsize=5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    # Move the title up
    plt.title(title, y=1.05)
    plt.xlabel('Model')
    plt.ylabel('Proportion of Unsafe Responses')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)


suite = "slack"
attack = "few-shot"
df = load_results(f"./results/{suite}/{attack}/results.jsonl")
plot_metrics(df, f"{suite}, {attack}")
# plt.tight_layout()
plt.show()