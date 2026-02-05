import os
import numpy as np
import tifffile
from tifffile import imread, imwrite
from tqdm import tqdm
import argparse
import glob


def split(src_dir, save_dir, patch_size=320, overlap_pix=20):
    os.makedirs(save_dir, exist_ok=True)
    stride = patch_size - overlap_pix
    if stride <= 0:
        raise ValueError('overlap_pix cannot exceed patch_size!')

    starts = lambda L: [k * stride if k * stride + patch_size <= L else L - patch_size
                        for k in range(int(np.ceil((L - patch_size) / stride)) + 1)]

    sample = imread(os.path.join(src_dir, '00.tif'))
    t, y_size, x_size = sample.shape
    dtype = sample.dtype
    y_starts = starts(y_size)
    x_starts = starts(x_size)
    print(f'共 {len(y_starts)} 行 × {len(x_starts)} 列 = '
          f'{len(y_starts) * len(x_starts)} 个窗口')
    name_num = len(glob.glob(f'{src_dir}/*.tif'))
    print(name_num)
    paths = [os.path.join(src_dir, f'{i:02d}.tif') for i in range(name_num)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    img_num = 0
    for path in tqdm(paths):
        print(img_num)
        print(path)
        vol = imread(path)
        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
                if yi == 5 and xi == 5:
                    a = 1
                win_slice = np.s_[:, y0:y0 + patch_size, x0:x0 + patch_size]
                out = vol[win_slice]
                out_folder = f'y{yi:02d}_x{xi:02d}'
                os.makedirs(f'{save_dir}/{out_folder}', exist_ok=True)
                tifffile.imwrite(f'{save_dir}/{out_folder}/{img_num}.tif', out.astype('uint8'))
                # if img_num == 0:
                #     with tifffile.TiffWriter(os.path.join(save_dir, out_name), bigtiff=True) as tif:
                #         tif.write(out)
                # else:
                #     with tifffile.TiffWriter(os.path.join(save_dir, out_name), append=True, bigtiff=True) as tif:
                #         tif.write(out)
        img_num += 1
        print(f'\nSplit {img_num}th file.')
    return len(y_starts), len(x_starts), y_size, x_size


if __name__ == '__main__':
    src_dir = 'F:/comet/MC/'
    save_dir = 'F:/comet/MC/patches'
    split(src_dir, save_dir)




