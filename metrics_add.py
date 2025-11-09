import json

# metrics_brats.json
# metrics_kits.json
with open('metrics.json', 'r') as f:
    data = json.load(f)

# Add the CR for the combined jp2k file to existing metrics
for case_name, modalities in data.items():
    for modality, slices in modalities.items():
        for slice_data in slices:
            cr_combined = slice_data['"compression_ratio_combined"']
            roi_size = slice_data['roi_only_bytes']
            bg_size = slice_data['bg_only_bytes']
            combined_size = slice_data['combined_bytes']
            
            # Calculate combined CR
            cr_combined_file = (cr_combined * (roi_size + bg_size)) / combined_size
            
            # Add to the data
            slice_data['compression_ratio_combined_file'] = cr_combined_file

# Save updated metrics
# metrics_updated_brats
# metrics_updated_kits
with open('metrics_updated.json', 'w') as f:
    json.dump(data, f, indent=2)