"""
result_plot.py 
--------------------------------------------
Author: Rui Shen, ChatGPT
- plot Box-whisker graph for overall compression ratio and roi compression ratio across algorithms for different dataset
- plot scatter plot shows relation between black space percentage, roi percentage, overall PSNR vs compression ratio
- plot a table showing overall CR, overall PSNR, Non-ROI CR, and ROI CR for different algorithms
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv('/Users/ruishen/Desktop/Main/CSComps_CompressionMedicalImages_2025Fall/all_metrics.csv')

# Replace inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

out_dir = "/Users/ruishen/Desktop/Main/CSComps_CompressionMedicalImages_2025Fall/plots"
os.makedirs(out_dir, exist_ok=True)

# Split datasets
df_brats = df[df["dataset"] == "brats"]
df_kits = df[df["dataset"] == "kits"]

# Compute global max values for consistent scaling
def get_global_max_values():
    """Compute maximum values across both datasets for consistent scaling"""
    
    # Overall compression ratio max
    overall_cr_max = max(
        df["cr_jpeg"].max() if "cr_jpeg" in df.columns else 0,
        df["compression_ratio_two_stream_jpeg2000"].max() if "compression_ratio_two_stream_jpeg2000" in df.columns else 0,
        df["compression_ratio_combined_spiht"].max() if "compression_ratio_combined_spiht" in df.columns else 0
    )
    
    # ROI compression ratio max
    roi_cr_max = max(
        df["cr_jpeg"].max() if "cr_jpeg" in df.columns else 0,  # JPEG uses same as overall
        df["roi_cr_jpeg2000"].max() if "roi_cr_jpeg2000" in df.columns else 0,
        df["roi_compression_ratio_spiht"].max() if "roi_compression_ratio_spiht" in df.columns else 0
    )
    
    # ROI percentage max
    roi_pct_max = max(
        df["roi_percentage"].max() if "roi_percentage" in df.columns else 0,
        df["roi_percentage_jpeg2000"].max() if "roi_percentage_jpeg2000" in df.columns else 0,
        df["roi_percentage_spiht"].max() if "roi_percentage_spiht" in df.columns else 0
    )
    
    # Black percentage max
    black_pct_max = df["black_percentage"].max() if "black_percentage" in df.columns else 100
    
    # PSNR max
    psnr_max = max(
        df["psnr_jpeg"].max() if "psnr_jpeg" in df.columns else 0,
        df["bg_psnr_jpeg2000"].max() if "bg_psnr_jpeg2000" in df.columns else 0,
        df["overall_psnr_spiht"].max() if "overall_psnr_spiht" in df.columns else 0
    )
    
    return {
        "overall_cr": overall_cr_max,
        "roi_cr": roi_cr_max,
        "roi_pct": roi_pct_max,
        "black_pct": black_pct_max,
        "psnr": psnr_max
    }

# Get global max values
global_maxes = get_global_max_values()

# Add padding for better visualization
for key in global_maxes:
    if not np.isnan(global_maxes[key]) and global_maxes[key] > 0:
        global_maxes[key] *= 1.1



# Helper: scatter with linear fit (lighter dots, darker line)
def scatter_with_fit(x, y, label, scatter_color, line_color, ax=None):
    if ax is None:
        ax = plt.gca()
    
    mask = (~x.isna()) & (~y.isna())
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)

    if len(x_clean) < 2:
        print(f"Not enough data points for linear fit for {label}")
        return
    
    # Light scatter points
    ax.scatter(
        x_clean,
        y_clean,
        s=15,
        alpha=0.25,
        c=scatter_color,
        marker='o'
    )

    # Linear fit
    coef = np.polyfit(x_clean, y_clean, 1)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = np.polyval(coef, x_line)

    # Darker, more visible fit line
    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=3,
        color=line_color,
        alpha=0.95,
        label=label
    )

# Main plotting function for one dataset
def make_plots_for_dataset(subset, dataset_name):
    """
    subset: rows belonging to a single dataset, e.g. df[df['dataset'] == 'brats']
    dataset_name: string used in titles/file names, e.g. 'BraTS' or 'KiTS'
    """

    # Boxplot: overall compression ratio
    jpeg_cr = subset["cr_jpeg"].dropna()
    j2k_cr = subset["compression_ratio_two_stream_jpeg2000"].dropna()
    spiht_cr = subset["compression_ratio_combined_spiht"].dropna()

    overall_medians = [
        np.median(jpeg_cr) if len(jpeg_cr) > 0 else np.nan,
        np.median(j2k_cr) if len(j2k_cr) > 0 else np.nan,
        np.median(spiht_cr) if len(spiht_cr) > 0 else np.nan,
    ]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.boxplot(
        [jpeg_cr, j2k_cr, spiht_cr],
        tick_labels=["JPEG", "JPEG 2000", "SPIHT"],
        showfliers=True,
    )
    ax.set_ylabel("Overall Compression Ratio")
    ax.set_title(f"Overall Compression Ratio by Algorithm ({dataset_name})")
    
    # Set y-axis limits starting from 0
    ax.set_ylim(0, global_maxes["overall_cr"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"overall_cr_{dataset_name}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    # Boxplot: ROI compression ratio
    jpeg_roi_mask = subset["has_roi"] == True
    jpeg_roi_cr = subset.loc[jpeg_roi_mask, "cr_jpeg"].dropna()

    j2k_roi_mask = subset["has_roi"] == True
    j2k_roi_cr = subset.loc[j2k_roi_mask, "roi_cr_jpeg2000"].dropna()

    spiht_roi_mask = subset["has_roi"] == True
    spiht_roi_cr = subset.loc[spiht_roi_mask, "roi_compression_ratio_spiht"].dropna()

    roi_medians = [
        np.median(jpeg_roi_cr) if len(jpeg_roi_cr) > 0 else np.nan,
        np.median(j2k_roi_cr) if len(j2k_roi_cr) > 0 else np.nan,
        np.median(spiht_roi_cr) if len(spiht_roi_cr) > 0 else np.nan,
    ]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.boxplot(
        [jpeg_roi_cr, j2k_roi_cr, spiht_roi_cr],
        tick_labels=["JPEG (ROI≈overall)", "JPEG 2000 ROI", "SPIHT ROI"],
        showfliers=True,
    )
    ax.set_ylabel("ROI Compression Ratio")
    ax.set_title(f"ROI Compression Ratio by Algorithm ({dataset_name})")
    
    # Set y-axis limits starting from 0
    ax.set_ylim(0, global_maxes["roi_cr"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"roi_cr_{dataset_name}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    # 3. Scatter and linear fit: ROI% vs overall CR
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # JPEG 2000
    j2k_x = subset.loc[j2k_roi_mask, "roi_percentage"]
    j2k_y = subset.loc[j2k_roi_mask, "compression_ratio_two_stream_jpeg2000"]
    scatter_with_fit(
        j2k_x,
        j2k_y,
        "JPEG 2000",
        scatter_color="orange",      
        line_color="saddlebrown",
        ax=ax
    )

    # SPIHT
    spiht_x = subset.loc[spiht_roi_mask, "roi_percentage"]
    spiht_y = subset.loc[spiht_roi_mask, "compression_ratio_combined_spiht"]
    scatter_with_fit(
        spiht_x,
        spiht_y,
        "SPIHT",
        scatter_color="lightgreen", 
        line_color="forestgreen",
        ax=ax
    )

    ax.set_xlabel("ROI Percentage")
    ax.set_ylabel("Overall Compression Ratio")
    ax.set_title(f"ROI % vs Overall Compression Ratio ({dataset_name})")
    
    # Set axis limits starting from 0
    ax.set_xlim(0, global_maxes["roi_pct"])
    ax.set_ylim(0, global_maxes["overall_cr"])
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"roi_vs_cr_{dataset_name}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


# Run for BraTS and KiTS
make_plots_for_dataset(df_brats, "BraTS")
make_plots_for_dataset(df_kits, "KiTS")

# Build summary metrics table

def compute_case_level_means(subset, algo):
    """
    subset = df for a single dataset+algorithm
    algo   = 'jpeg', 'jpeg2000', or 'spiht'

    Returns one row per *case* with averaged slice metrics.
    """

    metrics = {
        "jpeg": {
            "overall_cr": "cr_jpeg",
            "overall_psnr": "psnr_jpeg",
            "black_pct": "black_percentage",
            "roi_pct": None,
            "bg_cr": None,
            "roi_cr": None,
        },
        "jpeg2000": {
            "overall_cr": "compression_ratio_two_stream_jpeg2000",
            "overall_psnr": None,
            "black_pct": "black_percentage",
            "roi_pct": "roi_percentage_jpeg2000",
            "bg_cr": "bg_cr_jpeg2000",
            "roi_cr": "roi_cr_jpeg2000",
            "roi_pct": "roi_percentage_jpeg2000"
        },
        "spiht": {
            "overall_cr": "compression_ratio_combined_spiht",
            "overall_psnr": "overall_psnr_spiht",
            "black_pct": "black_percentage",
            "roi_pct": "roi_percentage_spiht",
            "bg_cr": "bg_compression_ratio_spiht",
            "roi_cr": "roi_compression_ratio_spiht",
            "roi_pct": "roi_percentage_spiht"
        }
    }

    met = metrics[algo]
    rows = []
    for case, case_df in subset.groupby("case"):
        row = {}
        # For each metric
        for name, col in met.items():
            if col is None or col not in case_df.columns:
                continue
            if name in ["roi_cr", "roi_pct"]:
                if "has_roi" in case_df.columns:
                    roi_df = case_df[case_df["has_roi"] == True]
                elif "has_roi" in case_df.columns:
                    roi_df = case_df[case_df["has_roi"] == True]
                else:
                    roi_df = pd.DataFrame()

                if len(roi_df) > 0:
                    row[name] = roi_df[col].mean()
                else:
                    row[name] = np.nan
            else:
                # Averaging across all slices
                row[name] = case_df[col].mean()

        rows.append((case, row))

    # Build dataframe indexed by case
    case_means = pd.DataFrame({case: data for case, data in rows}).T
    return case_means

def summarize_dataset(df_dataset, dataset_name):
    """
    Produces 3 rows: JPEG, JPEG2000, SPIHT averages for one dataset.
    """

    rows = []

    # JPEG
    jpeg_sub = df_dataset[df_dataset["cr_jpeg"].notna()]
    if len(jpeg_sub) > 0:
        jpeg_case_means = compute_case_level_means(jpeg_sub, "jpeg")
        row = jpeg_case_means.mean().to_dict()
        row["algorithm"] = "JPEG"
        row["dataset"] = dataset_name
        rows.append(row)

    # JPEG2000
    j2k_sub = df_dataset[df_dataset["compression_ratio_two_stream_jpeg2000"].notna()]
    if len(j2k_sub) > 0:
        j2k_case_means = compute_case_level_means(j2k_sub, "jpeg2000")
        row = j2k_case_means.mean().to_dict()
        row["algorithm"] = "JPEG2000"
        row["dataset"] = dataset_name
        rows.append(row)

    # SPIHT
    spiht_sub = df_dataset[df_dataset["compression_ratio_combined_spiht"].notna()]
    if len(spiht_sub) > 0:
        spiht_case_means = compute_case_level_means(spiht_sub, "spiht")
        row = spiht_case_means.mean().to_dict()
        row["algorithm"] = "SPIHT"
        row["dataset"] = dataset_name
        rows.append(row)

    return rows


# Create the final 6-row table
brats_rows = summarize_dataset(df_brats, "BraTS")
kits_rows  = summarize_dataset(df_kits, "KiTS")

summary_table = pd.DataFrame(brats_rows + kits_rows)

# Save table
summary_path = os.path.join(out_dir, "summary_table.csv")
summary_table.to_csv(summary_path, index=False)

print(summary_table)

# Reorder columns: dataset, algorithm, then metrics
cols_order = [
    "dataset",
    "algorithm",
    "overall_cr",
    "overall_psnr",
    "bg_cr",
    "roi_cr",
    "black_pct",
    "roi_pct"
]

# Keep only columns exist
cols_order = [c for c in cols_order if c in summary_table.columns]
summary_table = summary_table[cols_order]


def save_table_simple(df, filename):
    # Make a copy and round numbers
    df_display = df.copy().round(3).astype(str)

    fig, ax = plt.subplots(figsize=(12, 2 + 0.4 * len(df_display)))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")

        # Header row
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_fontsize(11)
            cell.set_text_props(weight="bold")

        # Left-align first two columns (dataset, algorithm) in data rows
        if row > 0 and col in [0, 1]:
            cell.set_text_props(ha="left")

    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved PNG table to:", out_path)

save_table_simple(summary_table, "summary_table_simple.png")

# Scatterplot: Overall CR vs Black Percentage (per dataset)
def scatter_black_vs_cr_per_dataset(df_dataset, dataset_name, filename):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # JPEG
    jpeg_mask = df_dataset["cr_jpeg"].notna()
    if jpeg_mask.any():
        ax.scatter(
            
            df_dataset.loc[jpeg_mask, "black_percentage"],
            df_dataset.loc[jpeg_mask, "cr_jpeg"],
            c="tab:blue",
            alpha=0.35,
            s=12,
            label="JPEG"
        )

    # JPEG 2000
    j2k_mask = df_dataset["compression_ratio_two_stream_jpeg2000"].notna()
    if j2k_mask.any():
        ax.scatter(
            
            df_dataset.loc[j2k_mask, "black_percentage"],
            df_dataset.loc[j2k_mask, "compression_ratio_two_stream_jpeg2000"],
            c="tab:orange",
            alpha=0.35,
            s=12,
            label="JPEG 2000"
        )

    # SPIHT
    spiht_mask = df_dataset["compression_ratio_combined_spiht"].notna()
    if spiht_mask.any():
        ax.scatter(
            
            df_dataset.loc[spiht_mask, "black_percentage"],
            df_dataset.loc[spiht_mask, "compression_ratio_combined_spiht"],
            c="tab:green",
            alpha=0.35,
            s=12,
            label="SPIHT"
        )

    ax.set_ylabel("Overall Compression Ratio")
    ax.set_xlabel("Black Space Percentage")
    ax.set_title(f"Overall CR vs Black % ({dataset_name})")
    
    # Set axis limits starting from 0
    ax.set_xlim(0, global_maxes["overall_cr"])
    ax.set_ylim(0, global_maxes["black_pct"])
    
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved scatterplot to:", out_path)


# Generate two separate plots
scatter_black_vs_cr_per_dataset(df_brats, "BraTS", "black_vs_cr_BraTS.png")
scatter_black_vs_cr_per_dataset(df_kits,  "KiTS",  "black_vs_cr_KiTS.png")

# Scatterplot: Overall CR vs Overall PSNR (per dataset)
def scatter_cr_vs_psnr_per_dataset(df_dataset, dataset_name, filename):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # JPEG: uses psnr_jpeg and cr_jpeg
    jpeg_mask = df_dataset["psnr_jpeg"].notna() & df_dataset["cr_jpeg"].notna()
    if jpeg_mask.any():
        ax.scatter(
            df_dataset.loc[jpeg_mask, "psnr_jpeg"],
            df_dataset.loc[jpeg_mask, "cr_jpeg"],
            c="tab:blue",
            alpha=0.35,
            s=12,
            label="JPEG"
        )

    # JPEG 2000: only plotted if an overall PSNR column exists in the CSV
    if "bg_psnr_jpeg2000" in df_dataset.columns:
        j2k_mask = df_dataset["bg_psnr_jpeg2000"].notna() & \
                   df_dataset["compression_ratio_two_stream_jpeg2000"].notna()
        if j2k_mask.any():
            ax.scatter(
                df_dataset.loc[j2k_mask, "bg_psnr_jpeg2000"],
                df_dataset.loc[j2k_mask, "compression_ratio_two_stream_jpeg2000"],
                c="tab:orange",
                alpha=0.35,
                s=12,
                label="JPEG 2000"
            )
    else:
        print(f"[{dataset_name}] No overall_psnr_jpeg2000 column; skipping JPEG 2000 in CR vs PSNR plot.")

    # SPIHT: uses overall_psnr_spiht and compression_ratio_combined_spiht
    if "overall_psnr_spiht" in df_dataset.columns:
        spiht_mask = df_dataset["overall_psnr_spiht"].notna() & \
                     df_dataset["compression_ratio_combined_spiht"].notna()
        if spiht_mask.any():
            ax.scatter(
                df_dataset.loc[spiht_mask, "overall_psnr_spiht"],
                df_dataset.loc[spiht_mask, "compression_ratio_combined_spiht"],
                c="tab:green",
                alpha=0.35,
                s=12,
                label="SPIHT"
            )

    ax.set_xlabel("Overall PSNR (dB)")
    ax.set_ylabel("Overall Compression Ratio")
    ax.set_title(f"Overall CR vs Overall PSNR ({dataset_name})")
    
    # Set axis limits starting from 0
    ax.set_xlim(0, global_maxes["psnr"])
    ax.set_ylim(0, global_maxes["overall_cr"])
    
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved CR vs PSNR scatterplot to:", out_path)


# Generate two separate CR–PSNR plots
scatter_cr_vs_psnr_per_dataset(df_brats, "BraTS", "cr_vs_psnr_BraTS.png")
scatter_cr_vs_psnr_per_dataset(df_kits,  "KiTS",  "cr_vs_psnr_KiTS.png")