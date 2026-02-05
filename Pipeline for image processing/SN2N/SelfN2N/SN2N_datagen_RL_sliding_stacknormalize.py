# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:26:30 2022

@author: quliying
"""


import os
import scipy.misc
import numpy as np
import tifffile
from glob import glob
import skimage.io as io
from skimage.io import imsave, imread, imshow
import skimage.transform as st
import random
# from .utils import utils
np.seterr(divide='ignore',invalid='ignore')


class data_generator():
    def __init__(self, img_path_left, img_path_right, save_path, pre_augment_mode = 0, augment_mode = 0, 
                    img_res = (128, 128), ifx2 = True, inter_method = 'Fourier', sliding_interval = 64):
        """
        SelfN2N data generator
        ------
        img_path
            path of raw images TO train
        save_path
            path of patched dataset TO save
        pre_augment_mode
            0: NONE; 
            1: direct interchange in t;
            2: interchange in single frame;
            3: interchange in multiple frame but in different regions
            {default: 0}
        augment_mode
            0: NONE; 
            1: double the dataset with random rotate&flip;
            2: eightfold the dataset with random rotate&flip;
            {default: 0}
        img_res
            patch size
            {default: (128, 128)}
        ifx2
            if re-scale TO original size
            True OR False
            {default: True}
        inter_method
            Scaling method
            'Fourier': Fourier re-scaling;
            'bilinear': spatial re-scaling;
            {default: 'Fourier'}
        
		------
        Example
        	DG = data_generator(img_path, save_path)
        	DG.savedata4folder_agument(threshold = 40)
        """
        self.pre_augment_mode = pre_augment_mode
        self.augment_mode = augment_mode
        self.img_path_left = img_path_left
        self.img_path_right = img_path_right
        self.save_path = save_path
        self.img_res = img_res
        self.ifx2 = ifx2
        self.inter_method = inter_method
        self.sliding_interval = sliding_interval
        
    def imgread_legacy(self, imgpath):
        """
        Not in use
        """
        img_stack = []
        img = imread(imgpath)
        img_stack.append(img)
        return img_stack

    def imread(self, imgpath):
        """
        Not in use
        """
        return scipy.misc.imread(imgpath).astype(np.float)
    
    def imread_stack(self, imgpath):
        image_stack = tifffile.imread(imgpath)
        return image_stack
    
    def block(self, image_stack):
        """
        Self-supervised data generator in xy
        ------
        image_stack
            image TO generate
        
        Returns
        -------
        left, right: noisy data pair
        """
        image_stack = image_stack.astype('float32')        
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        if image_stack.ndim == 1: 
            imsize = self.img_res
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
            left = np.zeros((1, imsize[0], imsize[1]))
            right = np.zeros((1, imsize[0], imsize[1]))
            return left, right
        [t, x, y] = image_stack.shape 
        image_stack
        upleft = []
        upright = []
        downright = []
        downleft = []
        for i in range(t):
            ul = image_stack[i, 0::2, 0::2]
            ur = image_stack[i, 0::2, 1::2]
            dr = image_stack[i, 1::2, 1::2]
            dl = image_stack[i, 1::2, 0::2]
            upleft.append(ul)
            upright.append(ur)
            downright.append(dr)
            downleft.append(dl)
        left = np.array(upleft) / 2 + np.array(downright) / 2
        right = np.array(upright) / 2 + np.array(downleft) / 2
        
        return left, right 
    
    def slidingWindow_RL(self, image_data_left, image_data_right, threshold_mode = 1, threshold = 15):
        """
        SelfN2N tool: patch
        ------
        image_data
            image TO generate
        interval
            interval pixel number to slide
            {default: 64}
        threshold (0 ~ 255)
            threshold to exclude some black patches
            {default: 15}
        
        Returns
        -------
        image_arr: patches with size (self.img_res)
        """
        interval = self.sliding_interval
        #image_data_left = 255*self.normalize(image_data_left)
        #image_data_right = 255*self.normalize(image_data_right)
        if threshold_mode == 1:
            threshold_real = threshold
        if threshold_mode == 2:
            avg = np.mean(image_data_right)
            threshold_real = avg+threshold
        bsize = self.img_res[0]
        image_arr_left = []
        image_arr_right = []
        #image_data = np.array(imread(img_path)).astype('float32')
        (h, w) = image_data_left.shape
        xx = int(np.floor(h - (bsize - interval)) / interval)
        yy = int(np.floor(w - (bsize - interval)) / interval)
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * interval
                right1 = (j - 1) * interval + bsize
                down = (i - 1) * interval
                up = (i - 1) * interval + bsize                    
                img_left = image_data_left[down:up, left1:right1]
                img_right = image_data_right[down:up, left1:right1]
                if np.sum(img_left) > bsize * bsize * (threshold_real) and np.sum(img_right) > bsize * bsize * (threshold_real):
                    image_arr_left.append(img_left)
                    image_arr_right.append(img_right)
                # if np.sum(img) > bsize * bsize * threshold:
                #     image_arr.append(img)
        image_arr_left = np.array(image_arr_left)
        image_arr_right = np.array(image_arr_right)
        return image_arr_left, image_arr_right
    
    def imwrite(self, image_path, image):
        """
        Not in use
        """
        image = image.astype('float32')
        image = image - np.min(image)
        image = 255 * image / np.max(image)
        img2save = image.astype('uint8')              
        imsave(image_path, img2save)
        

    
    def fourier_inter(self, image_stack):
        """
        SelfN2N tool: Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with 2x size 
        """
        imsize = self.img_res
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        [t, x, y] = image_stack.shape
        imgf1 = np.zeros((t, imsize[0], imsize[1]))
        
        for slice in range(t):
            img = image_stack[slice, :, :]
            imgsz = np.array([x, y])
            tem1 = np.divide(imgsz, 2)
            tem2 = np.multiply(tem1, 2)
            tem3 = np.subtract(imgsz, tem2)
            b = (tem3 == np.array([0, 0]))
            if b[0] == True:
                sz = imgsz - 1
            else:
                sz = imgsz            
            n = np.array([2, 2])
            ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
            ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
            idx = np.add(ttem1, ttem2)
            padsize = np.array([x/2, y/2], dtype = 'int')
            pad_wid = np.ceil(padsize[0]).astype('int')
            img = np.pad(img, ((pad_wid, 0), (pad_wid, 0)), 'symmetric')
            img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
            imgsz1 = np.array(img.shape)
            tttem1 = np.multiply(n, imgsz1)
            tttem2 = np.subtract(n, 1)
            newsz = np.round(np.subtract(tttem1, tttem2))
            img1 = self.interpft(img, newsz[0], 0)
            img1 = self.interpft(img1, newsz[1], 1)
            idx = idx.astype('int')
            ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
            ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
            imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
            imgf1[imgf1 < 0] = 0
        return imgf1
    
    def interpft(self, x, ny, dim = 0):
        '''
        Function to interpolate using FT method, based on matlab interpft()
        ------
        x 
            array for interpolation
        ny 
            length of returned vector post-interpolation
        dim
            performs interpolation along dimension DIM
            {default: 0}

        Returns
        -------
        y: interpolated data
        '''
    
        if dim >= 1: 
        #if interpolating along columns, dim = 1
            x = np.swapaxes(x,0,dim)
        #temporarily swap axes so calculations are universal regardless of dim
        if len(x.shape) == 1:            
        #interpolation should always happen along same axis ultimately
            x = np.expand_dims(x,axis=1)
    
        siz = x.shape
        [m, n] = x.shape
    
        a = np.fft.fft(x,m,0)
        nyqst = int(np.ceil((m+1)/2))
        b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)
    
        if np.remainder(m,2)==0:
            b[nyqst,:] = b[nyqst,:]/2
            b[nyqst+ny-m,:] = b[nyqst,:]
    
        y = np.fft.irfft(b,b.shape[0],0)
        y = y * ny / m
        y = np.reshape(y, [y.shape[0],siz[1]])
        y = np.squeeze(y)
    
        if dim >= 1:  
        #switches dimensions back here to get desired form
            y = np.swapaxes(y,0,dim)
    
        return y
    
    
    def data_augment(self, img_data, mode):    
        """
        SelfN2N tool: Random flip&rotate
        ------
        img_data
            image TO augmentation
        mode
            mode of flip&rotate

        Returns
        -------
        img_data: image after flip&rotate
        """
        if mode == 1: 
            img_data = np.flipud(np.rot90(img_data)) 
        elif mode == 2: 
            img_data = np.flipud(img_data) 
        elif mode == 3: 
            img_data = np.fliplr(img_data) 
        elif mode == 4: 
            img_data = np.fliplr(np.rot90(img_data))
        elif mode == 5: 
            img_data = np.rot90(img_data)
        elif mode == 6:
            img_data = np.rot90(img_data, k = 2)
        elif mode == 7: 
            img_data = np.rot90(img_data, k = 3)
        return img_data
    

    def random_interchange(self, imga, imgb = [], size = (64, 64), mode = 1):
        """
        SelfN2N tool: Random interchange
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        mode
            1: direct interchange (same ROI) in t axial
            2: interchange in single image
            3: interchange (two ROIs) in multiple images

        Returns
        -------
        img: image after ROI interchange
        """
        if mode == 0:
            return imga

        if imgb == []:
            mode = 2    
        if mode == 1: #interchange along t-axial 
            img = self.interchange_multiple(imga, imgb, size = size, ifdirect = False)
        elif mode == 2: #interchange in single image
            img = self.interchange_single(imga, size = size)
        elif mode == 3: #interchange in different images
            img = self.interchange_multiple(imga, imgb, size = size, ifdirect = False)

        return img

    def interchange_multiple(self, imga, imgb, size = (64, 64), ifdirect = False):
        """
        SelfN2N tool: Core of random interchange in multiple images
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        ifdirect
            interchange in same (True) or different (False) ROIs
            {default: False}

        Returns
        -------
        imga: image after ROI interchange
        """
        h = size[0]
        w = size[1]
        if h < np.min((np.size(imga, 0), np.size(imgb, 0))) and w < np.min((np.size(imga, 1), np.size(imgb, 1))):
            xa = random.randint(0, np.size(imga, 0) - h)
            ya = random.randint(0, np.size(imga, 1) - w)
            if ifdirect:
                imga[xa: xa + h, ya : ya + w] = imgb[xa: xa + h, ya : ya + w] 
            else:
                xb = random.randint(0, np.size(imgb, 0) - h)
                yb = random.randint(0, np.size(imgb, 1) - w)
                imga[xa: xa + h, ya : ya + w] = imgb[xb : xb + h, yb : yb + w]

        return imga

    def interchange_single(self, img, size = (64, 64)):
        """
        SelfN2N tool: Core of random interchange in single image
        ------
        img
            image TO ROI interchange
        size
            ROI size
            {default: (64, 64)}

        Returns
        -------
        img: image after ROI interchange
        """
        h = size[0]
        w = size[1]

        if h < np.size(img, 0) and w < np.size(img, 1):
            x1 = random.randint(0, np.size(img, 0) - h)
            y1 = random.randint(0, np.size(img, 1) - w)
            x2 = random.randint(0, np.size(img, 0) - h)
            y2 = random.randint(0, np.size(img, 1) - w)
            img[x1: x1 + h, y1 : y1 + w], img[x2 : x2 + h, y2 : y2 + w] = \
            img[x2 : x2 + h, y2 : y2 + w], img[x1: x1 + h, y1 : y1 + w]
        return img
    
    def random_interchange_RL(self, imga_left, imga_right, imgb_left = [], imgb_right = [], size = (64, 64), mode = 1):
        """
        SelfN2N tool: Random interchange
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        mode
            1: direct interchange (same ROI) in t axial
            2: interchange in single image
            3: interchange (two ROIs) in multiple images

        Returns
        -------
        img: image after ROI interchange
        """
        if mode == 0:
            return imga_left, imga_right

        #if imgb_left == []:
        #    mode = 2    
        if mode == 1: #interchange along t-axial 
            imga_left, imga_right = self.interchange_multiple_RL(imga_left, imga_right, imgb_left, imgb_right, size = size, ifdirect = False)
        elif mode == 2: #interchange in single image
            imga_left, imga_right = self.interchange_single_RL(imga_left, imga_right, size = size)
        elif mode == 3: #interchange in different images
            imga_left, imga_right = self.interchange_multiple_RL(imga_left, imga_right, imgb_left, imgb_right, size = size, ifdirect = False)

        return imga_left, imga_right
    
    def interchange_multiple_RL(self, imga_left, imga_right, imgb_left, imgb_right, size = (64, 64), ifdirect = False):
        """
        SelfN2N tool: Core of random interchange in multiple images
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        ifdirect
            interchange in same (True) or different (False) ROIs
            {default: False}

        Returns
        -------
        imga: image after ROI interchange
        """
        h = size[0]
        w = size[1]
        if h < np.min((np.size(imga_left, 0), np.size(imgb_left, 0))) and w < np.min((np.size(imga_left, 1), np.size(imgb_left, 1))):
            xa = random.randint(0, np.size(imga_left, 0) - h)
            ya = random.randint(0, np.size(imga_left, 1) - w)
            if ifdirect:
                imga_left[xa: xa + h, ya : ya + w] = imgb_left[xa: xa + h, ya : ya + w]
                imga_right[xa: xa + h, ya : ya + w] = imgb_right[xa: xa + h, ya : ya + w] 
            else:
                xb = random.randint(0, np.size(imgb_left, 0) - h)
                yb = random.randint(0, np.size(imgb_left, 1) - w)
                imga_left[xa: xa + h, ya : ya + w] = imgb_left[xb : xb + h, yb : yb + w]
                imga_right[xa: xa + h, ya : ya + w] = imgb_right[xb : xb + h, yb : yb + w]
        return imga_left, imga_right

    def interchange_single_RL(self, img_left, img_right, size = (64, 64)):
        """
        SelfN2N tool: Core of random interchange in single image
        ------
        img
            image TO ROI interchange
        size
            ROI size
            {default: (64, 64)}

        Returns
        -------
        img: image after ROI interchange
        """
        h = size[0]
        w = size[1]

        if h < np.size(img_left, 0) and w < np.size(img_left, 1):
            x1 = random.randint(0, np.size(img_left, 0) - h)
            y1 = random.randint(0, np.size(img_left, 1) - w)
            x2 = random.randint(0, np.size(img_left, 0) - h)
            y2 = random.randint(0, np.size(img_left, 1) - w)
            img_left[x1: x1 + h, y1 : y1 + w], img_left[x2 : x2 + h, y2 : y2 + w] = \
            img_left[x2 : x2 + h, y2 : y2 + w], img_left[x1: x1 + h, y1 : y1 + w]
            img_right[x1: x1 + h, y1 : y1 + w], img_right[x2 : x2 + h, y2 : y2 + w] = \
            img_right[x2 : x2 + h, y2 : y2 + w], img_right[x1: x1 + h, y1 : y1 + w]
        return img_left, img_right
    
    def normalize(self, stack):
        stack = stack.astype('float32')
        stack = stack - np.min(stack)
        stack = stack / np.max(stack)
        return stack
    
    def savedata_RL(self, left, right, flage):
        """
        SelfN2N tool: TO save data
            DATA structure: img, label (h, 2 x h) uint8
        ------
        image_stack
            data TO save
        flage  
            data number

        Returns
        -------
        NULL
        """
        
        # left, right = self.block(image_stack)            
        if self.ifx2:
            imsize = self.img_res
            if self.inter_method == 'bilinear':
                left = self.imgstack_resize(left, imsize)
                right = self.imgstack_resize(right, imsize)
            elif self.inter_method == 'Fourier':
                left = self.fourier_inter(left)
                right = self.fourier_inter(right)
        else:
            imsize = self.img_res
            # imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
        if np.ndim(left)==2:
            left = np.expand_dims(left, 0)
            right = np.expand_dims(right, 0)
            
        if np.ndim(left)==1:
            return flage
            
        [t, x, y] = left.shape
        size1 = (imsize[0], imsize[1] * 2)
        imgpart = np.zeros(size1, dtype = 'float32')
        imgpart_aug = np.zeros(size1, dtype = 'float32')
        imgpart_list = []
        for tt in range(t):
            temp_l = left[tt, :, :]
            #temp_l = self.normalize(temp_l)
            temp_r = right[tt, :, :]
            #temp_r = self.normalize(temp_r)
            imgpart[0 : imsize[0], 0 : imsize[1]] = temp_l        
            imgpart[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r
            imgpart_copy = imgpart.copy()
            imgpart_list.append(imgpart_copy)
            if self.augment_mode == 1:
                mode = random.randint(0, 7)                
                temp_l_aug = self.data_augment(temp_l, mode)
                temp_r_aug = self.data_augment(temp_r, mode)
                imgpart_aug[0 : imsize[0], 0 : imsize[1]] = temp_l_aug        
                imgpart_aug[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                imgpart_aug_copy = imgpart_aug.copy()
                imgpart_list.append(imgpart_aug_copy)
            elif self.augment_mode == 2:
                for m in range(1, 8):
                    temp_l_aug = self.data_augment(temp_l, m)
                    temp_r_aug = self.data_augment(temp_r, m)
                    imgpart_aug[0 : imsize[0], 0 : imsize[1]] = temp_l_aug       
                    imgpart_aug[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                    imgpart_aug_copy = imgpart_aug.copy()
                    imgpart_list.append(imgpart_aug_copy)          

        imgpart_list_copy = np.array(imgpart_list)
          
        [slices, x, y] = imgpart_list_copy.shape
        for s in range(slices):    
            img = imgpart_list_copy[s, :, :]
            img = img * 255
            if np.mean(img) > 0:                
                imsave(('%s/%d.tif') %(self.save_path, flage), img.astype('uint8'))
                flage = flage + 1
                if flage % 100 == 0:
                    print('Saving training images:', flage)
        return flage


    def savedata4folder_agument_RL(self, flage=1, interval=64, threshold_mode=2, threshold=15, 
                                size=(64, 64), times=1, roll=1):
        if self.pre_augment_mode == 0:
            times = 0
            roll = 0
        datapath_list_left = []
        datapath_list_right = []
        for (root, dirs, files) in os.walk(self.img_path_left):
            for j, Ufile in enumerate(files):
                img_path_left = os.path.join(root, Ufile)
                datapath_list_left.append(img_path_left)
                
        for (root, dirs, files) in os.walk(self.img_path_right):
            for j, Ufile in enumerate(files):
                img_path_right = os.path.join(root, Ufile)
                datapath_list_right.append(img_path_right)
                
        l = len(datapath_list_left)
        for ll in range(l):
            # 读取图像堆栈
            image_data_left = self.imread_stack(datapath_list_left[ll])
            image_data_right = self.imread_stack(datapath_list_right[ll])
            print('Processing number %d frame' % (ll + 1))

            # 对整个堆栈进行归一化
            image_data_left = self.normalize(image_data_left)
            image_data_right = self.normalize(image_data_right)

            try:
                [t, x, y] = image_data_left.shape
                for taxial in range(t):
                    # 原始数据保存
                    image_arr_left, image_arr_right = self.slidingWindow_RL(image_data_left[taxial, :, :], 
                                                                            image_data_right[taxial, :, :], 
                                                                            threshold_mode=threshold_mode, 
                                                                            threshold=threshold)
                    flage = self.savedata_RL(image_arr_left, image_arr_right, flage)

                    # 随机区域交换增强
                    for circlelarge in range(roll):
                        if times >= 1:
                            image_data_pre_left, image_data_pre_right = self.random_interchange_RL(imga_left=image_data_left[taxial, :, :],
                                                                                                imga_right=image_data_right[taxial, :, :], 
                                                                                                imgb_left=image_data_left[random.randint(0, t - 1), :, :], 
                                                                                                imgb_right=image_data_right[random.randint(0, t - 1), :, :], 
                                                                                                size=size, 
                                                                                                mode=self.pre_augment_mode)
                            # 重复增强 N-1 次
                            for circle in range(times - 1):
                                image_data_pre_left, image_data_pre_right = self.random_interchange_RL(imga_left=image_data_pre_left,
                                                                                                    imga_right=image_data_pre_right, 
                                                                                                    imgb_left=image_data_left[random.randint(0, t - 1), :, :], 
                                                                                                    imgb_right=image_data_right[random.randint(0, t - 1), :, :], 
                                                                                                    size=size, 
                                                                                                    mode=self.pre_augment_mode)
                            image_arr_left, image_arr_right = self.slidingWindow_RL(image_data_pre_left, 
                                                                                    image_data_pre_right, 
                                                                                    threshold_mode=threshold_mode, 
                                                                                    threshold=threshold)
                            flage = self.savedata_RL(image_arr_left, image_arr_right, flage)

            except ValueError:
                if self.pre_augment_mode == 3:
                    image_data_b_left = self.imread_stack(datapath_list_left[random.randint(0, l - 1)])
                    image_data_b_right = self.imread_stack(datapath_list_right[random.randint(0, l - 1)])
                else:
                    image_data_b_left = []
                    image_data_b_right = []
                # 原始数据保存
                image_arr_left, image_arr_right = self.slidingWindow_RL(image_data_left, 
                                                                        image_data_right, 
                                                                        threshold_mode=threshold_mode, 
                                                                        threshold=threshold)
                flage = self.savedata_RL(image_arr_left, image_arr_right, flage)
                
                for circlelarge in range(roll):
                    if times >= 1:
                        image_data_pre_left, image_data_pre_right = self.random_interchange_RL(imga_left=image_data_left,
                                                                                            imga_right=image_data_right, 
                                                                                            imgb_left=image_data_b_left, 
                                                                                            imgb_right=image_data_b_right, 
                                                                                            size=size, 
                                                                                            mode=self.pre_augment_mode)
                        # 重复增强 N-1 次
                        for circle in range(times - 1):
                            image_data_pre_left, image_data_pre_right = self.random_interchange_RL(imga_left=image_data_pre_left,
                                                                                                imga_right=image_data_pre_right, 
                                                                                                imgb_left=image_data_b_left, 
                                                                                                imgb_right=image_data_b_right, 
                                                                                                size=size, 
                                                                                                mode=self.pre_augment_mode)
                        image_arr_left, image_arr_right = self.slidingWindow_RL(image_data_pre_left, 
                                                                                image_data_pre_right, 
                                                                                threshold_mode=threshold_mode, 
                                                                                threshold=threshold)
                        flage = self.savedata_RL(image_arr_left, image_arr_right, flage)
        return

        
        



