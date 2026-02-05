import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import skimage.io as io
from skimage.io import imread, imsave, imshow
from skimage import transform
import os
from glob import glob
from torchvision import transforms
import torch
import torch.nn.functional as F
import skimage
from skimage import measure
import tifffile

def imread_stack(imgpath):
    image_stack = tifffile.imread(imgpath)
#        image_stack=image_stack.astype('float32')
#        [t,x,y]=image_stack.shape
#        imgmax=np.max(np.max(image_stack,1),1)
#        imgmax=imgmax.reshape(t,1,1)
#        image_stack=255*image_stack/imgmax
#        image_stack=image_stack.astype('uint8')
    return image_stack

def normalize_V1(stack):
    stack = stack.astype('float32')
    stack = stack - np.min(stack)
    stack = stack / np.max(stack)
    return stack

def normalize(stack):
    stack = stack.astype('float32')
    lim = np.zeros([len(stack), 2], dtype=stack[0].dtype)
    for i in range(len(stack)):
        lim[i] = stack[i].min(), stack[i].max()
        stack[i] -= lim[i, 0] 
        stack[i] = stack[i] / lim[i, 1]
    return stack

def normalize_percentage(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32): 
    """Percentile-based image normalization.""" 
    # dt = np.dtype('i8')
    mi = np.percentile(x,pmin,axis=axis,keepdims=True) #np.percantile对象可以是多维数组
    ma = np.percentile(x,pmax,axis=axis,keepdims=True) 
    # ma = ma
    # print(ma)
    # mi = np.array(mi, dt)
    
    # print(np.double(mi), ma)
    # print(np.dtype([mi,'i8']))
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype) 
    # return normalize_mi_ma_V2(x, mi, ma, clip=clip, eps=eps, dtype=dtype) 


def normalize_percentage_V2(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32): 
    """Percentile-based image normalization.""" 
    # dt = np.dtype('i8')
    mi = np.percentile(x,pmin,axis=axis,keepdims=True) #np.percantile对象可以是多维数组
    ma = np.percentile(x,pmax,axis=axis,keepdims=True) 
    # mi = np.array(mi, dt)
    
    # print(np.double(mi), ma)
    # print(np.dtype([mi,'i8']))
    return normalize_mi_ma_V2(x, mi, ma, clip=clip, eps=eps, dtype=dtype) 

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32): 
     if dtype is not None: 
         x = x.astype(dtype,copy=False) 
         mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False) 
         ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False) 
         eps = dtype(eps) 
     try: 
         import numexpr 
         x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )") 
     except ImportError: 
         x = (x - mi) / ( ma - mi + eps ) 
     if clip: 
        x = np.clip(x,0,1) 
        return x 
    
def normalize_mi_ma_V2(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32): 
     if dtype is not None: 
         x = x.astype(dtype,copy=False) 
         mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False) 
         ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False) 
         eps = dtype(eps) 
     try: 
         import numexpr 
         # x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )") 
         x = numexpr.evaluate("(x - mi) / ( ma )") 
     except ImportError: 
         # x = (x - mi) / ( ma - mi + eps ) 
         x = (x -mi) / ma
     if clip: 
        x = np.clip(x,0,1) 
        return x 

def normalize_minmse(x, target): 
     """Affine rescaling of x, such that the mean squared error to target is minimal.""" 
     cov = np.cov(x.flatten(),target.flatten()) 
     alpha = cov[0,1] / (cov[0,0]+1e-10) 
     beta = target.mean() - alpha*x.mean() 
     return alpha * x + beta
 

def caculate_metric(img_gt_path, img_pred_path, save_path,verbose = True, SaveTXT = False):    
    im_gt = imread(img_gt_path)
    if im_gt.ndim == 3:        
            im_gt = np.squeeze(im_gt)#t, x, y            
    im_gt = normalize(im_gt)
    im_pred = imread(img_pred_path)
    if im_pred.ndim == 3:        
            im_pred = np.squeeze(im_pred)#t, x, y   
    im_pred = normalize(im_pred)
    psnr = skimage.metrics.peak_signal_noise_ratio(im_gt, im_pred)
    ssim = skimage.metrics.structural_similarity(im_gt, im_pred)
    # mse = skimage.metrics.mean_squared_error(im_gt, im_pred)
    # rmse = math.sqrt(mse)
    nrmse = skimage.metrics.normalized_root_mse(im_gt, im_pred)
    if verbose:
        print("PSNR = %f" %(psnr)+'\r\n')
        print("SSIM = %f" %(ssim)+'\r\n')
        print("NRMSE = %f" %(nrmse)+'\r\n')
    # os.path.join(save_path, '/result.txt')
    if SaveTXT:
        f = open(save_path, 'w')
        f.write("PSNR = %f" %(psnr)+'\r\n')
        f.write("SSIM = %f" %(ssim)+'\r\n')
        f.write("NRMSE = %f" %(nrmse)+'\r\n')
        f.close()
    return
    