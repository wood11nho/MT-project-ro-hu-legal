import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
import numpy as np

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Professional Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") # Makes fonts bigger for PowerPoint
COLORS = ["#4C72B0", "#55A868", "#C44E52"] # Blue, Green, Red (Academic palette)

# Robust Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stopwords (Basic list to remove common words so we see "Legal" terms)
STOPWORDS_HU = set(["a", "az", "és", "hogy", "vagy", "nem", "is", "egy", "s", "azt"])
STOPWORDS_RO = set(["de", "și", "în", "la", "care", "cu", "o", "sa", "din", "pentru", "nu", "un", "fi"])

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def get_token_counts(text_series):
    """Returns a list of lengths (number of words) for each sentence."""
    return text_series.astype(str).apply(lambda x: len(x.split()))

def get_most_common_words(text_series, stopwords, n=15):
    """Finds top N most frequent words, excluding stopwords."""
    all_text = " ".join(text_series.astype(str).tolist()).lower()
    # Simple regex to keep only words
    words = re.findall(r'\w+', all_text)
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(filtered_words).most_common(n)

# ------------------------------------------------------------------------------
# Main Visualization Logic
# ------------------------------------------------------------------------------
def generate_visualizations():
    print("--- Generating Professional Visualizations ---")
    
    # 1. Load Data
    dfs = {}
    for split in ["train", "val", "test"]:
        file_path = PROCESSED_DIR / f"{split}.csv"
        if file_path.exists():
            print(f"Loading {split} set...")
            dfs[split] = pd.read_csv(file_path)
        else:
            print(f"Warning: {split}.csv not found. Skipping.")

    if not dfs:
        print("No data found! Please run prepare_splits.py first.")
        return

    # Combine for global stats (if multiple splits exist)
    full_df = pd.concat(dfs.values(), ignore_index=True)
    
    # -------------------------------------------------------
    # CHART 1: Dataset Split Distribution (Pie Chart)
    # -------------------------------------------------------
    if len(dfs) > 1:
        print("Generating Split Distribution Chart...")
        plt.figure(figsize=(10, 6))
        
        sizes = [len(df) for df in dfs.values()]
        labels = [f"{name.capitalize()}\n({size:,})" for name, size in zip(dfs.keys(), sizes)]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=COLORS)
        plt.title('Dataset Split Distribution (JRC-Acquis)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "1_dataset_splits.png", dpi=300)
        plt.close()

    # -------------------------------------------------------
    # CHART 2: Sentence Length Distribution (Histogram)
    # -------------------------------------------------------
    print("Generating Sentence Length Histogram...")
    
    hu_lengths = get_token_counts(full_df["hu"])
    ro_lengths = get_token_counts(full_df["ro"])

    plt.figure(figsize=(12, 6))
    sns.kdeplot(hu_lengths, fill=True, label="Hungarian (Source)", color=COLORS[0], clip=(0, 100))
    sns.kdeplot(ro_lengths, fill=True, label="Romanian (Target)", color=COLORS[1], clip=(0, 100))
    
    plt.title('Sentence Length Distribution (Density)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Tokens (Words)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 80) # Cut off extremely long tails for cleaner graph
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_length_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # CHART 3: Top Legal Terms (Bar Chart)
    # -------------------------------------------------------
    print("Generating Top Legal Terms Chart...")
    
    top_hu = get_most_common_words(full_df["hu"], STOPWORDS_HU)
    top_ro = get_most_common_words(full_df["ro"], STOPWORDS_RO)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Hungarian Plot
    y_pos = np.arange(len(top_hu))
    axes[0].barh(y_pos, [count for word, count in top_hu], align='center', color=COLORS[0])
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([word for word, count in top_hu])
    axes[0].invert_yaxis()  # Labels read top-to-bottom
    axes[0].set_title('Top 15 Hungarian Terms', fontweight='bold')

    # Romanian Plot
    y_pos = np.arange(len(top_ro))
    axes[1].barh(y_pos, [count for word, count in top_ro], align='center', color=COLORS[1])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([word for word, count in top_ro])
    axes[1].invert_yaxis()
    axes[1].set_title('Top 15 Romanian Terms', fontweight='bold')

    plt.suptitle('Most Frequent Legal Terminology (Stopwords Removed)', fontsize=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_top_terminology.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # CHART 4: Length Correlation (Scatter Plot)
    # -------------------------------------------------------
    print("Generating Alignment Correlation Chart...")
    
    # Sample 2000 points to keep the plot clean (don't plot 400k dots)
    sample_df = full_df.sample(min(2000, len(full_df)), random_state=42)
    sample_hu_len = get_token_counts(sample_df["hu"])
    sample_ro_len = get_token_counts(sample_df["ro"])

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=sample_hu_len, y=sample_ro_len, alpha=0.5, color=COLORS[2], edgecolor=None)
    
    # Add a diagonal line (perfect alignment)
    max_val = max(sample_hu_len.max(), sample_ro_len.max())
    plt.plot([0, max_val], [0, max_val], ls="--", c=".3", label="Perfect 1:1 Ratio")
    
    plt.title('Source vs. Target Length Correlation', fontsize=16, fontweight='bold')
    plt.xlabel('Hungarian Length (tokens)')
    plt.ylabel('Romanian Length (tokens)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_length_correlation.png", dpi=300)
    plt.close()

    # -------------------------------------------------------
    # TEXT: Summary Statistics Table
    # -------------------------------------------------------
    stats = {
        "Metric": ["Total Sentence Pairs", "Avg Source Length", "Avg Target Length", "Source Vocabulary (est.)", "Target Vocabulary (est.)"],
        "Value": [
            f"{len(full_df):,}",
            f"{hu_lengths.mean():.1f} words",
            f"{ro_lengths.mean():.1f} words",
            f"{len(set(' '.join(full_df['hu'].astype(str)).split())):,}",
            f"{len(set(' '.join(full_df['ro'].astype(str)).split())):,}"
        ]
    }
    stats_df = pd.DataFrame(stats)
    stats_path = OUTPUT_DIR / "dataset_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    
    print("\n--- Processing Complete! ---")
    print(f"Images saved to: {OUTPUT_DIR}")
    print(stats_df)

if __name__ == "__main__":
    generate_visualizations()