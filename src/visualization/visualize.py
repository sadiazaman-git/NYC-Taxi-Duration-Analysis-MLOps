import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use('ggplot')

# Path to save plots
save_plot_path = Path(__file__).parent / 'plots'
save_plot_path.mkdir(parents=True, exist_ok=True)

# Load metrics.json
metrics_path = Path(__file__).parent.parent.parent / 'metrics.json'
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Prepare data for the plot
models = ['random_forest', 'xgboost', 'gradient_boosting', 'linear_regression']
train_scores = [metrics[f'{model}_train_r2'] for model in models]
val_scores = [metrics[f'{model}_val_r2'] for model in models]

# Create a nested bar chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_scores, width, label='Train R2 Score')
bars2 = ax.bar(x + width/2, val_scores, width, label='Val R2 Score')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Models')
ax.set_ylabel('R2 Score')
ax.set_title('R2 Scores by Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Add labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height*100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

# Save the plot
plt.tight_layout()
plt.savefig(save_plot_path / 'bar_plot.png')
