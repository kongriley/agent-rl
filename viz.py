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

def plot_metrics(df):
    """Create a bar chart showing the proportion of unsafe responses."""
    plt.figure(figsize=(10, 6))
    
    # Calculate proportion of unsafe responses
    df['total'] = df['unsafe'] + df['safe']
    df['unsafe_ratio'] = df['unsafe'] / df['total']
    
    # Extract model names from the redteam_model column
    df['model'] = df['redteam_model'].apply(lambda x: x.split('/')[-1].replace('-Instruct', ''))
    
    # Create bar chart
    bars = plt.bar(df['model'], df['unsafe_ratio'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.title('Proportion of Unsafe Responses by Model')
    plt.xlabel('Model')
    plt.ylabel('Proportion of Unsafe Responses')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)

df = load_results("./results/workspace/zero-shot/results.jsonl")
plot_metrics(df)
plt.tight_layout()
plt.show()