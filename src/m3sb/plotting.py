import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_radar_chart(
    results_df: pd.DataFrame,
    title: str
):
    
    methods = results_df['merge_method'].unique().tolist()
    datasets = results_df['dataset'].unique().tolist()
    num_vars = len(datasets)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method in methods:
        method_data = results_df[results_df['merge_method'] == method]
        values = method_data.set_index('dataset').loc[datasets]['accuracy'].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_rlabel_position(0)
    ax.set_thetagrids(np.degrees(angles[:-1]), datasets)
    ax.set_ylim(0, 100)
    
    plt.title(title, size=14, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

def plot_grouped_bar_chart(
    methods: list[str],
    results_data: dict[str, list[float]],
    title: str
):
    experiment_names = list(results_data.keys())
    num_methods = len(methods)
    num_experiments = len(experiment_names)

    bar_width = 0.25
    x = np.arange(num_experiments)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, method in enumerate(methods):
        accuracies = [results[i] for results in results_data.values()]
        offset = bar_width * i
        bars = ax.bar(x + offset, accuracies, width=bar_width, label=method)
        ax.bar_label(bars, fmt='%.2f%%', padding=3)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(experiment_names)
    ax.legend()
    
    all_values = [item for sublist in results_data.values() for item in sublist]
    if all_values:
        min_val = min(v for v in all_values if v > 0)
        ax.set_ylim([max(0, min_val - 10), 105])

    fig.tight_layout()
    plt.show()