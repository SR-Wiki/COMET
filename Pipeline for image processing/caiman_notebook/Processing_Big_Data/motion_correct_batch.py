import time
import cv2
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import os.path
import logging
import tifffile
import shutil
import os
import argparse

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
import glob


def motion_correct():
    # ----------------------------Parameters for batch processing-------------------------------------
    # modify the input_folder and output_folder
    # files in the input_folder should be named as "i.tif", where i is the sequence number
    input_folder = 'F:/comet/MC/\\'
    output_folder = 'F:/comet/MC/MC\\'
    segment_length = 400     # size of each segment, choose a large one while avoiding memory issues
    overlap_size = 50        # overlap between each segment (>= 20)
    step = segment_length - overlap_size

    tmp_folder = input_folder + 'tmp\\'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(tmp_folder, exist_ok=True)
    os.makedirs(tmp_folder + 'MC_segment', exist_ok=True)
    input_tif_path = []
    n_processes = os.cpu_count()  # 使用所有可用的CPU核心
    print('n_processes:', n_processes)

    max_shifts = (128, 128)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides = (64, 64)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (32, 32)  # overlap between patches (size of patch strides+overlaps)
    max_deviation_rigid = 2  # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

    # %% start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    img_path_list = glob.glob(input_folder + '*.tif')
    print(img_path_list)
    num_images = len(img_path_list)

    for index in range(num_images):
        start_time = time.time()
        img_path = input_folder + f'{index}.tif'
        print(f'Reading {img_path}')
        im = tifffile.imread(img_path)
        img_name = img_path.replace(input_folder, '')
        img_name = img_name.replace('.tif', '')
        print(img_name)

        # ---------------------------------------------------------------------- #
        # splitting and rolling operations for very long temporal sequence
        # adjust according to your system RAM
        # for time efficiency, set a large segment_length which fits for RAM
        # it is recommended that overlap_size >= 20
        t = im.shape[0]
        if index == 0:
            num_segments = int(np.ceil(t / step))
        else:
            # may need modeifications
            if (t + overlap_size) % step == 0:
                num_segments = int(np.ceil((t + overlap_size) / step))
            else:
                num_segments = int(np.ceil((t + overlap_size) / step)) - 1
        print(f'Total length: {t}')
        print(f'Number of Segments: {num_segments}')
        start_frame = 0
        for i in range(num_segments):
            end_frame = start_frame + segment_length if start_frame + segment_length < t else t
            if index >= 1 and i == 0:
                end_frame -= overlap_size
            print(f'Current Segment: [{start_frame}, {end_frame}]')
            tmp_segment = im[start_frame:end_frame, :, :]
            current_segment = im[start_frame:end_frame, :, :]
            # add overlap to register different segements from the same or adjacent images
            # if index == 1 and i == 0:
            #     output_name = '0_700-1000_MC.tif'
            if i > 0:
                print(f'Loading previous segment from {tmp_folder}MC_segment/{output_name}')
                pre_segment = tifffile.imread(tmp_folder + 'MC_segment/' + output_name)
                current_segment = tmp_segment
                current_segment[:overlap_size, :, :] = pre_segment[-overlap_size:, :, :]
            if index >= 1 and i == 0:
                print(f'Loading previous segment from {tmp_folder}MC_segment/{output_name}')
                pre_segment = tifffile.imread(tmp_folder + 'MC_segment/' + output_name)
                current_segment = np.concatenate((pre_segment[-overlap_size:, :, :], tmp_segment), axis=0)

            print(current_segment.shape)

            tmp_path = tmp_folder + 'MC_segment/' + img_name + f'_{start_frame}-{end_frame}.tif'
            tifffile.imwrite(tmp_path, current_segment)
            # create a motion correction object
            mc = MotionCorrect(tmp_path, dview=dview, max_shifts=max_shifts,
                               strides=strides, overlaps=overlaps,
                               max_deviation_rigid=max_deviation_rigid,
                               shifts_opencv=shifts_opencv, nonneg_movie=True,
                               border_nan=border_nan)

            # correct for rigid motion correction and save the file (in memory mapped form)
            mc.motion_correct(save_movie=True)
            # load motion corrected movie
            m_rig = cm.load(mc.mmap_file)
            bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
            mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
            mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
            mc.motion_correct(save_movie=True, template=mc.total_template_rig)
            output_name = img_name + f'_{start_frame}-{end_frame}_MC.tif'

            m_els = mc.fname_tot_els
            # 加载内存映射文件
            # mmap_data = cm.load('/home/lintao/caiman_data/temp/Substack (1-500)_els__d1_512_d2_512_d3_1_order_F_frames_500.mmap')

            fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
            F_frames = cm.load(fname_mc[0], outtype=np.float16)
            images = F_frames.T.reshape((current_segment.shape[0],) + (F_frames.shape[1], F_frames.shape[2]),
                                        order='F')
            if index >= 1 and i == 0:
                images = images[overlap_size:, :, :]
            print(images.shape)
            for frame in range(images.shape[0]):
                c_frame = images[frame, :, :]
                images[frame, :, :] = ((c_frame - c_frame.min()) / (c_frame.max() - c_frame.min())) * 255
            # F_frames = ((F_frames - F_frames.min()) / (F_frames.max() - F_frames.min())) * 255
            tifffile.imwrite(tmp_folder + 'MC_segment/' + output_name, F_frames.astype(np.uint8))
            print(f'Save to {tmp_folder}MC_segment/{output_name}')
            if index >= 1 and i == 0:
                start_frame += step
                start_frame -= overlap_size
            else:
                start_frame += step

        result = np.zeros_like(im)
        start_frame = 0
        for i in range(num_segments):
            end_frame = start_frame + segment_length if start_frame + segment_length < t else t
            if index >= 1 and i == 0:
                end_frame -= overlap_size
            output_name = img_name + f'_{start_frame}-{end_frame}_MC.tif'
            current_segment = tifffile.imread(tmp_folder + 'MC_segment/' + output_name)
            if index >= 1 and i == 0:
                result[start_frame:end_frame, :, :] = current_segment[overlap_size:, :, :]
            else:
                result[start_frame:end_frame, :, :] = current_segment
            # 删除caiman临时内存（可注释掉）

            if index >= 1 and i == 0:
                start_frame += step
                start_frame -= overlap_size
            else:
                start_frame += step

        tifffile.imwrite(output_folder + img_name + '_MC.tif', result.astype('uint8'))
        print('Save final motion-corrected image to ' + output_folder + img_name + '_MC.tif')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Total time cost: {elapsed_time // 60} minutes')

    if 'dview' in locals():
        cm.stop_server(dview=dview)

    # remove temp files
    for index in range(num_images):
        start_time = time.time()
        img_path = input_folder + f'{index}.tif'
        print(f'Reading {img_path}')
        im = tifffile.imread(img_path)
        img_name = img_path.replace(input_folder, '')
        img_name = img_name.replace('.tif', '')
        print(img_name)

        segment_length = 400
        overlap_size = 50
        step = segment_length - overlap_size
        t = im.shape[0]
        if index == 0:
            num_segments = int(np.ceil(t / step))
        else:
            # may need modeifications
            if (t + overlap_size) % step == 0:
                num_segments = int(np.ceil((t + overlap_size) / step))
            else:
                num_segments = int(np.ceil((t + overlap_size) / step)) - 1
        start_frame = 0
        for i in range(num_segments):
            end_frame = start_frame + segment_length if start_frame + segment_length < t else t
            if index >= 1 and i == 0:
                end_frame -= overlap_size
            try:
                os.remove(
                    f'C:/Users/User/caiman_data/temp/{img_name}_{start_frame}-{end_frame}_rig__d1_{im.shape[1]}_d2_{im.shape[1]}_d3_1_order_F_frames_{end_frame - start_frame}.mmap')
            except Exception:
                print('Unable to delete mmap file.')
            if index >= 1 and i == 0:
                start_frame += step
                start_frame -= overlap_size
            else:
                start_frame += step
    try:
        shutil.rmtree(tmp_folder)
    except Exception:
        print('Unable to remove temp folder.')


if __name__ == '__main__':
    motion_correct()








