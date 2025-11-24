import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv( '/Users/ruishen/Desktop/Main/CSComps_CompressionMedicalImages_2025Fall/all_metrics.csv')

# Replace inf with NaN to avoid issues
df = df.replace([np.inf, -np.inf], np.nan)


# Helper: scatter with linear fit
def scatter_with_fit(x, y, label, color):
    mask = (~x.isna()) & (~y.isna())
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)

    if len(x_clean) < 2:
        print(f"Not enough data points for linear fit for {label}")
        return
    
    # Small dots — use 'c=' to avoid confusion
    plt.scatter(
        x_clean,
        y_clean,
        s=8,
        alpha=0.4,
        c=color,        # MUST be a valid color name
        marker='o'
    )

    # Linear fit
    coef = np.polyfit(x_clean, y_clean, 1)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = np.polyval(coef, x_line)

    plt.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=3,   # bold line
        color=color,
        alpha=0.9,
        label=f"{label}"
    )


# Main plotting function for one dataset
def make_plots_for_dataset(subset, dataset_name):
    """
    subset: rows belonging to a single dataset, e.g. df[df['dataset'] == 'brats']
    dataset_name: string used in titles, e.g. 'BraTS' or 'KiTS'
    """

    # -----------------------------
    # 1. Boxplot: overall compression ratio
    # -----------------------------
    jpeg_cr = subset["cr_jpeg"].dropna()
    j2k_cr = subset["compression_ratio_combined_jpeg2000"].dropna()
    spiht_cr = subset["compression_ratio_combined_spiht"].dropna()

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [jpeg_cr, j2k_cr, spiht_cr],
        labels=["JPEG", "JPEG 2000", "SPIHT"],
        showfliers=True,
    )
    plt.ylabel("Overall Compression Ratio")
    plt.title(f"Overall Compression Ratio by Algorithm ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"Overall Compression Ratio by Algorithm ({dataset_name})", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Boxplot: ROI compression ratio
    #    JPEG ROI CR ≈ overall CR for slices that have ROI (using has_roi_jpeg2000)
    #    JPEG 2000 ROI CR: roi_cr_jpeg2000
    #    SPIHT ROI CR: roi_compression_ratio_spiht

    jpeg_roi_mask = subset["has_roi_jpeg2000"] == True
    jpeg_roi_cr = subset.loc[jpeg_roi_mask, "cr_jpeg"].dropna()

    j2k_roi_mask = subset["has_roi_jpeg2000"] == True
    j2k_roi_cr = subset.loc[j2k_roi_mask, "roi_cr_jpeg2000"].dropna()

    spiht_roi_mask = subset["has_roi_spiht"] == True
    spiht_roi_cr = subset.loc[spiht_roi_mask, "roi_compression_ratio_spiht"].dropna()

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [jpeg_roi_cr, j2k_roi_cr, spiht_roi_cr],
        labels=["JPEG (ROI≈overall)", "JPEG 2000 ROI", "SPIHT ROI"],
        showfliers=True,
    )
    plt.ylabel("ROI Compression Ratio")
    plt.title(f"ROI Compression Ratio by Algorithm ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"ROI Compression Ratio by Algorithm ({dataset_name})", dpi=300, bbox_inches='tight')
    plt.show()

 
    # 3. Scatter + linear fit:
    #    x-axis: ROI percentage
    #    y-axis: overall compression ratio
    #
    #    JPEG:  x = roi_percentage_jpeg2000, y = cr_jpeg
    #    J2K:   x = roi_percentage_jpeg2000, y = compression_ratio_combined_jpeg2000
    #    SPIHT: x = roi_percentage_spiht,    y = compression_ratio_combined_spiht
    #    (All restricted to this dataset.)

    plt.figure(figsize=(7, 5))




    # JPEG 2000
    j2k_x = subset.loc[j2k_roi_mask, "roi_percentage_jpeg2000"]
    j2k_y = subset.loc[j2k_roi_mask, "compression_ratio_combined_jpeg2000"]
    scatter_with_fit(j2k_x, j2k_y, "JPEG 2000", "tab:orange")

    # SPIHT
    spiht_x = subset.loc[spiht_roi_mask, "roi_percentage_spiht"]
    spiht_y = subset.loc[spiht_roi_mask, "compression_ratio_combined_spiht"]
    scatter_with_fit(spiht_x, spiht_y, "SPIHT", "tab:green")


    plt.xlabel("ROI Percentage")
    plt.ylabel("Overall Compression Ratio")
    plt.title(f"ROI % vs Overall Compression Ratio ({dataset_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ROI % vs Overall Compression Ratio ({dataset_name})", dpi=300, bbox_inches='tight')
    plt.show()


# Make plots for BraTS and KiTS

df_brats = df[df["dataset"] == "brats"]
df_kits  = df[df["dataset"] == "kits"]

make_plots_for_dataset(df_brats, "BraTS")
make_plots_for_dataset(df_kits, "KiTS")
