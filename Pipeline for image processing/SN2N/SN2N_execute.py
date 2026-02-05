import os
import numpy as np
from glob import glob
from SelfN2N.SN2N_datagen_RL_sliding_stacknormalize import data_generator
from SelfN2N.selfn2n_nonormalize import SelfN2N
from pathlib import Path
import argparse
import tifffile as tiff
import torch
from SelfN2N.testmodel_Sparse import AUnet
import sys
from inference import predict_


def split_file(src: Path, even_path, odd_path, test_path):
    print(f'Processing {src.name} ...')
    try:
        with tiff.TiffFile(str(src)) as tif:
            frames = tif.asarray()
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]
    except Exception as e:
        print(f'  ├─ Failed to read {src.name}: {e}', file=sys.stderr)
        return

    first_frame = frames[0]
    test_out = test_path / f'{src.stem}_first.tif'
    tiff.imwrite(str(test_out), first_frame, photometric='minisblack')
    print(f'  ├─ First frame -> {test_out.name}')

    left_frames = frames[0::2]
    right_frames = frames[1::2]

    if left_frames.size:
        left_out = even_path / f'{src.stem}_left.tif'
        tiff.imwrite(str(left_out), left_frames, photometric='minisblack')
        print(f'  ├─ {left_frames.shape[0]} left frames -> {left_out.name}')
    if right_frames.size:
        right_out = odd_path / f'{src.stem}_right.tif'
        tiff.imwrite(str(right_out), right_frames, photometric='minisblack')
        print(f'  └─ {right_frames.shape[0]} right frames -> {right_out.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--img_path',
                        default='../rawdata',
                        type=str,
                        required=False,
                        help='Path to the raw images for training.')
    parser.add_argument('--P2Pmode',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        help="""Patch2Patch augmentation mode. 
                                0: None; 
                                1: Direct interchange in time; 
                                2: Interchange in a single frame; 
                                3: Interchange in multiple frames but in different regions.""")
    parser.add_argument('--P2Pup',
                        type=int,
                        default=1,
                        help='Increase the dataset to its (1 + P2Pup) times size.')
    parser.add_argument('--BAmode',
                        type=int,
                        default=0,
                        choices=[0, 1, 2],
                        help="""Basic augmentation mode. 
                                0: None; 
                                1: Double the dataset with random rotate and flip; 
                                2: Eightfold the dataset with random rotate and flip.""")
    parser.add_argument('--SWsize',
                        type=int,
                        default=64,
                        help='Interval in pixels of the sliding window for generating image patches.')
    parser.add_argument('--bs',
                        type=int,
                        default=64,
                        help='training batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-4,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of epochs')
    parser.add_argument('--sn2n_loss',
                        type=float,
                        default=1,
                        help='Weight of self-constrained loss')
    parser.add_argument('--final_model',
                        type=bool,
                        default=True,
                        help='if only inference with the final model')

    # data generation
    args = parser.parse_args()
    img_path = Path(args.img_path)
    P2Pmode = args.P2Pmode
    P2Pup = args.P2Pup
    BAmode = args.BAmode
    SWsize = args.SWsize
    bs = args.bs
    lr = args.lr
    test_batch_size =1
    epochs = args.epochs
    reg = args.sn2n_loss
    final_model = args.final_model

    save_folder = img_path / 'generated_data'
    os.makedirs(save_folder, exist_ok=True)

    left_path = save_folder / 'odd'
    right_path = save_folder / 'even'
    save_path = save_folder / 'DL_datasets'
    test_path = save_folder / 'test_data'
    os.makedirs(left_path, exist_ok=True)
    os.makedirs(right_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # generate even-odd frames
    tif_paths = list(img_path.glob('*.tif')) + list(img_path.glob('*.tiff'))
    if not tif_paths:
        print('No .tif/.tiff files found in ../rawdata/')
    for p in tif_paths:
        split_file(p, even_path=left_path, odd_path=right_path, test_path=test_path)

    # generate dataset
    tif_paths = list(save_path.glob('*.tif')) + list(img_path.glob('*.tiff'))
    if len(tif_paths) < 1000:
        d = data_generator(img_path_left=left_path, img_path_right=right_path, save_path=save_path,
                           pre_augment_mode=P2Pmode,
                           augment_mode=BAmode, img_res=(128, 128), ifx2=False, inter_method='Fourier',
                           sliding_interval=32)
        d.savedata4folder_agument_RL(interval=SWsize, times=P2Pup, roll=1, threshold_mode=2, threshold=0)
        print('dataset has been generated')
    else:
        print('Training dataset already generated, skip this step.')


    # model training
    datasets_name = ''
    reg_sparse = 0
    prefix = 'Neuron_'

    tests_name = (prefix + (
            '%s_EPOCH%d_BS%d_LOSSconsis_%.2f_LOSSsparse%.2f_' % (datasets_name, epochs, bs, reg, reg_sparse)))

    torch.cuda.empty_cache()
    os.makedirs(str(img_path) + '/images/%s' % tests_name, exist_ok=True)
    os.makedirs(str(img_path) + '/images/%s/weights' % tests_name, exist_ok=True)
    os.makedirs(str(img_path) + '/images/%s/images' % tests_name, exist_ok=True)

    sn2nunet = SelfN2N(dataset_name=datasets_name, tests_name=tests_name, reg=reg, reg_sparse=reg_sparse,
                       constrained_type='L1', lr=lr, epochs=epochs, train_batch_size=bs,
                       ifadaptive_lr=False, test_batch_size=test_batch_size, img_path=str(img_path))
    sn2nunet.train()


    # inference
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = Path(str(img_path) + f'/images/{tests_name}/weights/')
    print(model_dir)
    if final_model:
        print('Inference with the last model...')
        model_list = sorted(model_dir.glob('*full.pth'))
    else:
        print('Inference with all models...')
        model_list = sorted(model_dir.glob('*.pth'))
    print(len(model_list))
    save_dir = img_path / ('images/%s/images' % tests_name)
    os.makedirs(save_dir, exist_ok=True)

    tif_list = sorted(Path(img_path).glob('*.tif'))

    if not tif_list or not model_list:
        print('No .tif or .pth found!')

    for pth_file in model_list:
        print(f'==== Loading model: {pth_file.name} ====')
        model = AUnet(n_channels=1, n_classes=1, bilinear=True).to(device)
        model.load_state_dict(torch.load(pth_file, map_location=device, weights_only=True))
        model.eval()

        for tif_file in tif_list:
            print(f'  --> infer {tif_file.name}')
            predict_(model, tif_file.parent, save_dir, tif_file.name, pth_file.name, ifGPU=True)

    print('Completed.')


