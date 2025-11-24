import os
import pandas as pd
import json
import numpy as np

ALGORITHM_METRIC_COLUMNS = ['mse', 'psnr', 'encoding_time', 'decoding_time', 'cr', 'overall_mse', 'overall_psnr', 'bg_cr', 'bg_mse', 'bg_psnr', 'roi_cr', 'roi_mse', 'roi_psnr', 'compression_ratio_combined', 'roi_compression_ratio', 'bg_compression_ratio', 'compressed_size_bytes', 'roi_compressed_bytes', 'bg_compressed_bytes', 'meta_bytes', 'compression_ratio_two_stream']

IMAGE_STATISTIC_COLUMNS = ['has_roi', 'roi_percentage', 'black_percentage', 'bg_pixel_count', 'bg_original_bytes', 'bg_only_bytes', 'roi_pixel_count', 'roi_original_bytes', 'roi_only_bytes', 'combined_bytes', 'original_size_bytes']

IMAGE_INDEXING_COLUMNS = ['dataset', 'case', 'slice_idx']



def mergeJPEGMetrics(dataset):
    all_data = {}
    for root, _, files in os.walk(os.path.join('outputs', dataset, 'metric_data', 'JPEG'), topdown=True):
        case = root.split('/')[-1]

        case_slices = []
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    slice_data = json.load(f)
                    slice_data['slice_idx'] = int(file.split('_')[1].split('.')[0])
                    case_slices.append(slice_data)
                    if slice_data['PSNR'] == None:
                        slice_data['PSNR'] = np.inf
            except Exception:
                continue

        if len(case_slices) != 0:
            all_data[case] = {'modality': 't1ce', 'slices': case_slices}
    
    with open(os.path.join('metrics', f'jpeg_{dataset}.json'), 'w') as f:
        json.dump(all_data, f)


def getDFFromJSON(algorithm, dataset):
    slice_data_key_name = 'slices'
    if algorithm == 'jpeg2000' and dataset == 'brats':
        slice_data_key_name = 't1ce'
    if algorithm == 'jpeg2000' and dataset == 'kits':
        slice_data_key_name = 'ct'
    
    with open(os.path.join('metrics', f'{algorithm}_{dataset}.json'), 'r') as f:
        all_data = json.load(f)
    
    case_dfs = []
    for case_name, case_data in all_data.items():
        if algorithm == 'spiht' and dataset == 'brats' and case_data['modality'] != 't1ce':
            continue
        
        case_data = {metric.lower(): [slice[metric] for slice in case_data[slice_data_key_name]] for metric in case_data[slice_data_key_name][0]}
        num_slices = len(case_data['slice_idx'])
        
        case_data['case'] = [case_name] * num_slices
        case_data['dataset'] = [dataset] * num_slices

        case_dfs.append(pd.DataFrame(case_data))
    
    df = pd.concat(case_dfs, ignore_index=True)
    df = df.rename(columns={orig: f'{orig}_{algorithm}' for orig in ALGORITHM_METRIC_COLUMNS})
    df = df.rename(columns={orig: f'{orig}_{algorithm}' for orig in IMAGE_STATISTIC_COLUMNS})
    return df

           
def combineAllMetrics():
    full_df = pd.DataFrame()
    for dataset in ('brats', 'kits'):
        dataset_df = pd.DataFrame()

        for algorithm in ('jpeg', 'jpeg2000', 'spiht'):
            algorithm_df = getDFFromJSON(algorithm, dataset)

            if len(dataset_df) == 0:
                dataset_df = algorithm_df.copy()
            else:
                dataset_df = dataset_df.merge(algorithm_df, how='inner', on=IMAGE_INDEXING_COLUMNS)

        if len(full_df) == 0:
            full_df = dataset_df.copy()
        else:
            full_df = pd.concat([full_df, dataset_df], ignore_index=True, sort=False)

    for statistic in IMAGE_STATISTIC_COLUMNS:
        cols_with_stat = [col_name for col_name in list(full_df.columns) if statistic in col_name]
        match len(cols_with_stat):
            case 1:
                full_df = full_df.rename(columns={cols_with_stat[0] : statistic})
            case 2:
                first_col, second_col = cols_with_stat
                if len(full_df[full_df[first_col] != full_df[second_col]]) == 0:
                    full_df = full_df.rename(columns={first_col : statistic}).drop(columns=[second_col])
            case 3:
                pass
            case _:
                print('Error - invalid columns')
                exit()
    
    full_df.to_csv('all_metrics.csv', index=False)
    

def main():
    # mergeJPEGMetrics('kits')
    mergeJPEGMetrics('brats')
    combineAllMetrics()


if __name__ == '__main__':
    main()