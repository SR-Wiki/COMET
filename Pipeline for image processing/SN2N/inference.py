import argparse
import torch
import numpy as np
import tifffile as tiff
import os
from pathlib import Path
from SelfN2N.testmodel_Sparse import AUnet


def predict_(model, img_path, save_path, fname, model_name, ifGPU=True):
    device = torch.device("cuda" if torch.cuda.is_available() and ifGPU else "cpu")

    image_data = tiff.imread(str(img_path / fname))
    if image_data.dtype != np.uint8:
        max_val = np.iinfo(image_data.dtype).max
        image_data = (image_data.astype(np.float32) * 255.0 / max_val).clip(0, 255).astype(np.uint8)
    try:
        t, x, y = image_data.shape
        image_data = image_data/255
        image_data = image_data.astype('float32')
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    except ValueError:
        t = 1
        x, y = image_data.shape
        image_data = image_data/255
        image_data = image_data.astype('float32')
        image_data = np.expand_dims(image_data, axis=0)  # 添加帧维度
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    for frame in range(t):
        single_frame = image_data[frame, :, :]
        #single_frame = normalize_(single_frame)
        single_frame = single_frame.astype('float32')
        single_frame = single_frame.reshape(1,x,y)
        imsize = (1,1,x,y)
        imagA = np.zeros(imsize)
        imagA[0,:,:,:] = single_frame
        imagA = torch.from_numpy(imagA)
        imagA = imagA.to(device, dtype = torch.float32)
        with torch.no_grad():
            test_pred = model(imagA)
        test_pred = test_pred.to(torch.device("cpu"))
        test_pred = test_pred.numpy()
        #test_pred = normalize_(test_pred)
        #plt.imshow(test_pred[0,0,:,:])
        #plt.show()

        test_pred_np[frame, :, :] = test_pred
        if frame % 100 == 0:
            print('predict %d images'% frame)
    #test_pred_np = normalize_(test_pred_np)

    os.makedirs(save_path, exist_ok=True)
    tiff.imwrite(os.path.join(save_path, f'{fname}_{model_name}.tif'), test_pred_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--img_path',
                        type=str,
                        required=True,
                        help='Path to the raw images for training.')
    parser.add_argument('--model_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to the trained model.')

    parser.add_argument('--save_path',
                        default=None,
                        type=str,
                        required=True,
                        help='folder to save inference results.')

    args = parser.parse_args()
    img_path = Path(args.img_path)
    model_path = Path(args.model_path)
    save_path = Path(args.save_path)

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(model_path)
    os.makedirs(save_path, exist_ok=True)
    tif_list = sorted(Path(img_path).glob('*.tif'))
    print(f'==== Loading model: {str(model_path)} ====')
    model = AUnet(n_channels=1, n_classes=1, bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    for tif_file in tif_list:
        print(f'  --> infer {tif_file.name}')
        predict_(model, tif_file.parent, save_path, tif_file.name, model_path.name, ifGPU=True)

    print('Inference Completed.')
