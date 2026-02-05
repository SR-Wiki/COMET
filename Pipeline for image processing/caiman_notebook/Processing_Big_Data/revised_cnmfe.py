if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str,
                        required=True,
                        help='Path to the folder conataining the full session.')
    parser.add_argument('--split_save_dir',
                        type=str,
                        required=True,
                        help='Path to the folder saving the split patches.')
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Path to the folder saving the split patches.')
    parser.add_argument('--patch_size',
                        type=int,
                        default=320,
                        required=False,
                        help='size of the split small patches')
    parser.add_argument('--overlap_pix',
                        type=int,
                        default=20,
                        required=False,
                        help='overlap between adjacent patches')
    parser.add_argument('--caiman_data_dir',
                        type=str,
                        default='F:\comet',
                        required=False,
                        help='folder to save caiman temp files')
    args = parser.parse_args()
    caiman_data_dir = args.caiman_data_dir
    os.environ['CAIMAN_DATA'] = caiman_data_dir

    # split to smaller patches
    from split_patch import split
    src_dir = args.src_dir
    split_save_dir = args.split_save_dir
    patch_size = args.patch_size
    overlap_pix = args.overlap_pix
    save_dir = args.save_dir
    os.makedirs(split_save_dir, exist_ok=True)

    y, x, H, W = split(src_dir, split_save_dir, patch_size=patch_size, overlap_pix=overlap_pix)
    print('y',y)
    print('x',x)
    print('H',H)
    print('W',W)

    # perform CNMF-E on each patch
    from source_extraction import cnmfe
    cnmfe(y, x, split_save_dir)

    # merge 逐行+逐列+最终合并
    from merge_patch_combined import merge
    
    merge(y, x, split_save_dir, patch_size, overlap_pix, save_dir, H, W)


