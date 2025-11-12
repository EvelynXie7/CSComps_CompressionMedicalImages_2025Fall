    # with open("/Users/justinvaughn/data/metrics/metrics_brats.json", 'r') as brats:
    #     data_brats = json.load(brats)

    # with open("/Users/justinvaughn/data/metrics/metrics_kits.json", 'r') as kits:
    #     data_kits = json.load(kits)
    
    # print("brats:")
    # # Calculate and print statistics
    # stats_brats = calculate_statistics(data_brats)
    # print_statistics(stats_brats)
    # print("kits:")
    # stats_kits = calculate_statistics(data_kits)
    # print_statistics(stats_kits)

    # # Create the graphs
    # black_pct_brats, comp_ratios_brats = extract_cr_vs_black_space(data_brats)
    # black_pct_kits, comp_ratios_kits = extract_cr_vs_black_space(data_kits)
    # plot_cr_vs_black_space(black_pct_brats, comp_ratios_brats, save_path='cr_vs_black_space_brats.png', dataset= "Brats")
    # plot_cr_vs_black_space(black_pct_kits, comp_ratios_kits, save_path='cr_vs_black_space_kits.png', dataset="Kits")