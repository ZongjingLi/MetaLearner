import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

def plot_single_experiment(dir_path):
    """
    Load ALL CSV files in a directory (each CSV = one experiment) and plot polished accuracy curves.
    - Each experiment has unique color pair (train = solid, test = dashed)
    - Train/test for same experiment share base color (distinct line styles)
    """
    # 1. Validate directory & find all CSVs
    if not os.path.isdir(dir_path):
        print(f"❌ Error: Directory not found at {dir_path}")
        return
    
    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))]
    if not csv_files:
        print(f"❌ No CSV files found in {dir_path}")
        return
    print(f"✅ Found {len(csv_files)} experiment CSV(s): {csv_files}")

    # 2. Define professional color palette (distinct for each experiment)
    # Use high-contrast, accessible colors (avoid red/green for color blindness)
    color_palette = [
        '#2E86AB',  # Deep blue (exp 1)
        '#F18F01',  # Orange (exp 2)
        '#C73E1D',  # Red (exp 3)
        '#6A994E',  # Green (exp 4)
        '#7209B7',  # Purple (exp 5)
        '#F72585'   # Pink (exp 6)
    ]
    # Cycle colors if more than 6 experiments
    exp_colors = {csv_files[i]: color_palette[i % len(color_palette)] for i in range(len(csv_files))}

    # 3. Set global plot style (clean + professional)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 4. Plot each experiment's train/test curves
    for csv_file in csv_files:
        csv_path = os.path.join(dir_path, csv_file)
        df = pd.read_csv(csv_path)
        exp_name = os.path.splitext(csv_file)[0]  # Use filename as experiment name
        color = exp_colors[csv_file]

        # Plot TRAIN accuracy (solid line + circle markers)
        ax.plot(
            df["epoch"], df["train_acc"],
            label=f"{exp_name} (Train)",
            color=color,
            marker='o', markersize=5, markeredgecolor='white', markeredgewidth=1,
            linewidth=2.2, alpha=0.9
        )
        # Plot TEST accuracy (dashed line + square markers, same base color)
        ax.plot(
            df["epoch"], df["test_acc"],
            label=f"{exp_name} (Test)",
            color=color,
            marker='s', markersize=5, markeredgecolor='white', markeredgewidth=1,
            linewidth=2.2, linestyle='--', alpha=0.9
        )

    # 5. Polished plot customizations
    # Axis labels (bold + readable)
    ax.set_xlabel("Epoch", fontsize=14, fontweight='medium', labelpad=12)
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='medium', labelpad=12)
    # Title
    ax.set_title("Experime" \
    "nt Accuracy Curves (All Runs)", fontsize=17, fontweight='bold', pad=25)
    # Legend (outside plot to avoid overlap)
    ax.legend(
        fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1),
        frameon=True, fancybox=True, shadow=True, framealpha=0.9
    )
    # Y-axis range (focus on accuracy)
    ax.set_ylim(bottom=0.5, top=1.02)
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Ticks
    ax.tick_params(axis='both', labelsize=12, pad=6)

    # 6. Save & show (high-DPI)
    plot_save_path = os.path.join(dir_path, "all_experiments_plot.png")
    plt.tight_layout()  # Adjust for legend
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Styled plot saved to: {plot_save_path}")
    plt.show()


if __name__ == "__main__":
    plot_single_experiment("outputs/logs/expr_1024")