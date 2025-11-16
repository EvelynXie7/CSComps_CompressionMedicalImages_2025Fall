import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def calculate_statistics(data: Dict, max_cases: int = None) -> Dict[str, Dict[str,float]]:
    """
    Caculate mean and standard deviation for compression ratio, bg_MSE, and bg_PSNR in brats
    """
    metrics={
        "compression_ratio_two_stream": [],
        "bg_mse": [],
        "bg_psnr": []
    }

    # Iterate through brats cases, and modalities
    for case_idx, (case_name, case_data) in enumerate(data.items()):
        if max_cases is not None and case_idx >=max_cases:
            break
        for modality, slices in case_data.items():
            for slice_data in slices:
                metrics['compression_ratio_two_stream'].append(slice_data['compression_ratio_two_stream'])
                metrics['bg_mse'].append(slice_data['bg_mse'])
                metrics['bg_psnr'].append(slice_data['bg_psnr'])

    #calculate stastics
    results={}
    for metric_name, values in metrics.items():
        results[metric_name]={
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "q1": np.percentile(values,25), 
            "median": np.median(values),
            "q3": np.percentile(values,75),
            "max": np.max(values),
            "iqr": np.percentile(values,75) - np.percentile(values,25),            
            "count": len(values)
        }
    
    return results

def extract_cr_vs_black_space(data: Dict, max_cases: int = None) -> Tuple[List[float], List[float]]:
    """
    Get compression ratio and black space for scatter plot
    """
    black_percentages = []
    compression_ratios = []
    
    for case_idx, (case_name, case_data) in enumerate(data.items()):
        if max_cases is not None and case_idx >=max_cases:
            break
        for modality, slices in case_data.items():
            for slice_data in slices:
                black_percentages.append(slice_data['black_percentage'])
                compression_ratios.append(slice_data['compression_ratio_two_stream'])
    
    return black_percentages, compression_ratios

def extract_cr_vs_roi_space(data: Dict, max_cases: int=None) -> Tuple[List[float], List[float]]:
    """
    Get compression ratio and black space for scatter plot
    """
    roi_percentages = []
    compression_ratios = []
    
    for case_idx, (case_name, case_data) in enumerate(data.items()):
        if max_cases is not None and case_idx >=max_cases:
            break
        for modality, slices in case_data.items():
            for slice_data in slices:
                roi_percentages.append(slice_data['roi_percentage'])
                compression_ratios.append(slice_data['compression_ratio_two_stream'])
    
    return roi_percentages, compression_ratios

def plot_cr_vs_black_space(black_percentages: List[float], 
                           compression_ratios: List[float],
                           save_path: str = None,
                           dataset: str= None) -> None:
    """
    Create scatter plot of compression ratio vs black space percentage.
    
    Args:
        black_percentages: List of black space percentages
        compression_ratios: List of compression ratios
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(black_percentages, compression_ratios, alpha=0.5, s=20)
    plt.xlabel('ROI Space (%)')
    plt.ylabel('Compression Ratio (Two-Stream)')
    plt.title(f"Compression Ratio vs ROI Percentage for {dataset}")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cr_vs_roi_space(roi_percentages: List[float], 
                           compression_ratios: List[float],
                           save_path: str = None,
                           dataset: str= None) -> None:
    """
    Create scatter plot of compression ratio vs black space percentage.
    
    Args:
        black_percentages: List of black space percentages
        compression_ratios: List of compression ratios
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(roi_percentages, compression_ratios, alpha=0.5, s=20)
    plt.xlabel('ROI Space (%)')
    plt.ylabel('Compression Ratio (Two-Stream)')
    plt.title(f"Compression Ratio vs ROI Percentage for {dataset}")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_statistics(stats: Dict) -> None:
    """Pretty print statistics results."""
    print("COMPRESSION STATISTICS")
    for metric, values in stats.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  Std Dev: {values['std']:.4f}")
        print(f"  Count: {values['count']}")
        print(f"  Min: {values['min']:.4f}")
        print(f"  q1: {values['q1']:.4f}")
        print(f"  Median: {values['median']:.4f}")
        print(f"  q3: {values['q3']:.4f}")
        print(f"  Maximum: {values['max']:.4f}") 
        print(f"  IQR: {values['iqr']:.4f}")         
        print(f"  Count: {values['count']}")
        """
            "min": np.min(values),
            "q1": np.percentile(values,25), 
            "median": np.median(values),
            "q3": np.percentile(values,75),
            "max": np.max(values),
            "iqr": np.percentile(values,75) - np.percentile(values,25),
        """

if __name__ == "__main__":
    # Load JSON data
    with open("/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats/metrics_brats.json", 'r') as brats:
        data_brats = json.load(brats)

    with open("/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits/metrics_kits.json", 'r') as kits:
        data_kits = json.load(kits)
    
    print("brats:")
    # Calculate and print statistics
    stats_brats = calculate_statistics(data_brats, max_cases=None)
    print_statistics(stats_brats)
    print("kits:")
    stats_kits = calculate_statistics(data_kits, max_cases=None)
    print_statistics(stats_kits)

    # Create the graphs
    black_pct_brats, comp_ratios_brats = extract_cr_vs_black_space(data_brats, max_cases=None)
    # black_pct_kits, comp_ratios_kits = extract_cr_vs_black_space(data_kits, max_cases=None)
    plot_cr_vs_black_space(black_pct_brats, comp_ratios_brats, save_path='cr_vs_black_space_brats.png', dataset= "Brats")
    # plot_cr_vs_black_space(black_pct_kits, comp_ratios_kits, save_path='cr_vs_black_space_kits.png', dataset="Kits")

    roi_pct_brats, comp_ratios_brats_roi =extract_cr_vs_roi_space(data_brats)
    # roi_pct_kits, comp_ratios_kits_roi = extract_cr_vs_roi_space(data_kits)
    plot_cr_vs_roi_space(roi_pct_brats, comp_ratios_brats_roi, save_path='cr_vs_roi_space_brats.png', dataset= "Brats")
    # plot_cr_vs_roi_space(roi_pct_kits, comp_ratios_kits_roi, save_path='cr_vs_roi_space_kits.png', dataset="Kits")
