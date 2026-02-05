import os
import time

import numpy as np
import psutil
import tifffile
import tifffile as tf
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
import glob
from matplotlib import pyplot as plt
import pickle
# os.environ['CAIMAN_DATA'] = 'F:\\comet'
import argparse


def cnmfe(y_num, x_num, split_save_dir):
    n_processes = os.cpu_count()  # 使用所有可用的CPU核心
    # n_processes = int(n_processes / 2)
    print('n_processes:', n_processes)

    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing',
                                                       n_processes=n_processes,
                                                       ignore_preexisting=False)
    print(n_processes)
    print(f"Successfully initilialized multicore processing with a pool of {n_processes} CPU cores")

    # parameters for source extraction and deconvolution
    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array([2, 2])  # expected half-width of neurons in pixels
    gSiz = 2 * gSig + 1  # half-width of bounding box created around neurons during initialization
    merge_thr = .7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20  # amount of overlap between the patches in pixels
    tsub = 2  # downsampling factor in time for initialization, increase if you have memory problems
    ssub = 1  # downsampling factor in space for initialization, increase if you have memory problems
    gnb = 0  # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None  # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = 0  # number of background components (rank) per patch (0 for CNMFE)
    min_corr = .8  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
    bord_px = 0

    parameters = params.CNMFParams(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                                'K': K,
                                                'gSig': gSig,
                                                'gSiz': gSiz,
                                                'merge_thr': merge_thr,
                                                'p': p,
                                                'tsub': tsub,
                                                'ssub': ssub,
                                                'rf': rf,
                                                'stride': stride_cnmf,
                                                'only_init': True,  # set it to True to run CNMF-E
                                                'nb': gnb,
                                                'nb_patch': nb_patch,
                                                'method_deconvolution': 'oasis',
                                                # could use 'cvxpy' alternatively
                                                'low_rank_background': low_rank_background,
                                                'update_background_components': True,
                                                # sometimes setting to False improve the results
                                                'min_corr': min_corr,
                                                'min_pnr': min_pnr,
                                                'normalize_init': False,  # just leave as is
                                                'center_psf': True,  # True for 1p
                                                'ssub_B': ssub_B,
                                                'ring_size_factor': ring_size_factor,
                                                'del_duplicates': True,
                                                # whether to remove duplicates from initialization
                                                'border_pix': bord_px,
                                                # number of pixels to not consider in the borders)
                                                'memory_fact': 1})

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str,
                        required=True,
                        help='Parent folder of split sessions.')
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

    for y in range(y_num):
        for x in range(x_num):
            input_path = f'{split_save_dir}/y{y:02d}_x{x:02d}\\'
            output_path = f'{split_save_dir}/y{y:02d}_x{x:02d}/cnmfe\\'
            os.makedirs(output_path, exist_ok=True)
            print(f'Now Processing folder {input_path}')

            img_path_list = glob.glob(input_path + '*.tif')
            print(len(img_path_list))
            img_name = f'y{y:02d}_x{x:02d}'

            print('Now do the memory mapping...')
            start_time = time.time()
            fname_new = cm.save_memmap(img_path_list, base_name=f'{img_name}_memmap', order='C',
                                       border_to_0=0, dview=cluster)
            print(fname_new)
            # 删除中间产生的内存映射文件，防止存储占用过大，可能需要根据系统更改上方的CAIMAN_DATA路径形式
            tmp_files = glob.glob('F:\\comet\\temp\\*.mmap')
            print(len(tmp_files))
            for file in tmp_files:
                try:
                    if file != fname_new:
                        os.remove(file)
                except Exception:
                    continue
            Yr, dims, T = cm.load_memmap(fname_new)
            images = Yr.T.reshape((T,) + dims, order='F')
            print('Complete memory mapping.')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Memory mapping costs {elapsed_time // 60} minutes')
            print(f'memmap image shape: {images.shape}')
            if images is None:
                raise ValueError("Images data is None")
            if np.isnan(images).any() or np.isinf(images).any():
                images = np.nan_to_num(images)  # 替换 NaN 和 Inf 值

            cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=2, swap_dim=False, center_psf=True)
            name = output_path + 'correlation_images_sigma2.tif'
            cn_threshold = cn_filter.copy()
            cn_threshold[cn_threshold < 0] = 0
            tifffile.imwrite(name, cn_threshold)

            name = output_path + 'pnr_images_sigma2.tif'
            # cm.summary_images.save_summary_images(pnr,name)
            tifffile.imwrite(name, pnr)

            
            cnm = cnmf.CNMF(n_processes=n_processes, params=parameters)
            print('CNMF CLASS CREATED')

            np.seterr(divide='ignore', invalid='ignore')
            print('Before Fitting')
            cnm.fit(images)
            print('Fitting completed.')
            cnm.estimates.Cn = cn_filter
            # cnm.save(output_results_path)
            try:
                os.remove(fname_new)
            except PermissionError as e:
                print(f"Error deleting file: {e}")

            cnm.estimates.plot_contours_nb(img=pnr)

            # high threshold
            r_values_min = 0.9  # threshold on space consistency (if you lower more components
            rval_lowest = -0.5
            SNR_lowest = 3
            cnn_lowest = 0.8
            # high threshold
            min_SNR = 4
            min_cnn_thr = 1
            cnm.params.set('quality',
                           {'rval_lowest': rval_lowest, 'SNR_lowest': SNR_lowest, 'min_SNR': min_SNR,
                            'cnn_lowest': cnn_lowest,
                            'rval_thr': r_values_min, 'min_cnn_thr': min_cnn_thr,
                            'use_cnn': False, 'use_ecc': True, 'max_ecc': 1.8, 'gSig_range': (2.5, 3)})
            # cnm.estimates.evaluate_components(images, cnm.params, dview=cluster)
            #
            # print('*****')
            # print(f"Total number of components: {len(cnm.estimates.C)}")
            # print(f"Number accepted: {len(cnm.estimates.idx_components)}")
            # print(f"Number rejected: {len(cnm.estimates.idx_components_bad)}")
            print(f'C shape: {cnm.estimates.C.shape}')
            #
            # idx_accepted = cnm.estimates.idx_components
            # all_contour_coords = [cnm.estimates.coordinates[idx]['coordinates'] for idx in idx_accepted]
            # idx_to_plot = 30
            # component_number = idx_accepted[idx_to_plot]
            # component_contour = all_contour_coords[idx_to_plot]
            # component_footprint = np.reshape(cnm.estimates.A[:, component_number].toarray(), dims, order='F')
            # plt.figure()
            # plt.imshow(component_footprint, cmap='gray')
            # plt.plot(component_contour[:, 0],
            #          component_contour[:, 1],
            #          color='pink',
            #          linewidth=2)
            # plt.title(f'Footprint/Contour {component_number}')
            # plt.savefig(output_path + img_name + '.png')
            # plt.close()
            # print(f"Number accepted: {len(cnm.estimates.idx_components)}")
            # print(f"Number rejected: {len(cnm.estimates.idx_components_bad)}")

            name = output_path + img_name + f'_cnm_estimates.pkl'
            with open(name, 'wb') as file:
                pickle.dump(cnm.estimates, file)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Total time cost: {elapsed_time // 60} minutes')


if __name__ == '__main__':
    cnmfe()









