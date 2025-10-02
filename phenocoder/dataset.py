from phenocoder.generator import DatasetGenerator

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('dir_input', type=str)
    parser.add_argument('dir_output', type=str)
    parser.add_argument('--dir_segmented', type=str, default=None)
    parser.add_argument('--qc_path', type=str, default=None)
    parser.add_argument('--patch_mode', type=str, default='grid')
    parser.add_argument('--channels', type=str, nargs='+', default=['01', '02', '03', '04'])
    parser.add_argument('--sampling_frac', type=float, default=None)
    parser.add_argument('--n_patches', type=int, default=200000)
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--mode', default='client')
    parser.add_argument('--port', default=52162)
    args = parser.parse_args()

    dataset_generator = DatasetGenerator(args.dir_input, args.dir_output, max_workers=args.max_workers, dir_segmented=args.dir_segmented, mode=args.patch_mode, channels=args.channels)
    dataset_generator.generate_dataset(sampling_frac=args.sampling_frac, n_patches=args.n_patches, n_bins=args.n_bins, per_channel=True, qc_path=args.qc_path)
    dataset_generator.save_stats()
