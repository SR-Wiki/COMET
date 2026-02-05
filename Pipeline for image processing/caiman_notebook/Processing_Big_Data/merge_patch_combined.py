#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并特定的四个区块（y0x0, y0x1, y1x0, y1x1）成 1944×1944 大图
"""
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import caiman as cm
import caiman as cm
import os, pickle
from scipy.sparse import csr_matrix, hstack, vstack
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
# from caiman.source_extraction.cnmf.merging import merge_components  # 把前面脚本保存为同名文件
from caiman.cluster import setup_cluster
from pathlib import Path
import logging
import numpy as np
import scipy
from scipy.sparse import csgraph, csc_matrix, lil_matrix, csr_matrix

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from caiman.source_extraction.cnmf.spatial import update_spatial_components, threshold_components
from caiman.source_extraction.cnmf.temporal import update_temporal_components
from caiman.source_extraction.cnmf.utilities import update_order_greedy
from caiman.source_extraction.cnmf.estimates import Estimates
from scipy.sparse import lil_matrix
from scipy.sparse import hstack as sparse_hstack


def merge_components(Y, A, b, C, R, f, S, sn_pix, temporal_params,
                     spatial_params, dview=None, thr=0.85, fast_merge=True,
                     mx=1000, bl=None, c1=None, sn=None, g=None,
                     merge_parallel=False) -> tuple[
    scipy.sparse.csc_matrix, np.ndarray, int, list, np.ndarray, float, float, float, float, list, np.ndarray]:
    """ Merging of spatially overlapping components that have highly correlated temporal activity

    The correlation threshold for merging overlapping components is user specified in thr

    Args:
        Y: np.ndarray
            residual movie after subtracting all found components
            (Y_res = Y - A*C - b*f) (d x T)

        A: sparse matrix
            matrix of spatial components (d x K)

        b: np.ndarray
             spatial background (vector of length d)

        C: np.ndarray
             matrix of temporal components (K x T)

        R: np.ndarray
             array of residuals (K x T)

        f:     np.ndarray
             temporal background (vector of length T)

        S:     np.ndarray
             matrix of deconvolved activity (spikes) (K x T)

        sn_pix: ndarray
             noise standard deviation for each pixel

        temporal_params: dictionary
             all the parameters that can be passed to the
             update_temporal_components function

        spatial_params: dictionary
             all the parameters that can be passed to the
             update_spatial_components function

        thr:   scalar between 0 and 1
             correlation threshold for merging (default 0.85)

        mx:    int
             maximum number of merging operations (default 50)

        sn_pix:    nd.array
             noise level for each pixel (vector of length d)

        fast_merge: bool
            if true perform rank 1 merging, otherwise takes best neuron

        bl:
             baseline for fluorescence trace for each row in C
        c1:
             initial concentration for each row in C
        g:
             discrete time constant for each row in C
        sn:
             noise level for each row in C

        merge_parallel: bool
             perform merging in parallel

    Returns:
        A:     sparse matrix
                matrix of merged spatial components (d x K)

        C:     np.ndarray
                matrix of merged temporal components (K x T)

        nr:    int
            number of components after merging

        merged_ROIs: list
            index of components that have been merged

        S:     np.ndarray
                matrix of merged deconvolved activity (spikes) (K x T)

        bl: float
            baseline for fluorescence trace

        c1: float
            initial concentration

        sn: float
            noise level

        g:  float
            discrete time constant

        empty: list
            indices of neurons that were removed, as they were merged with other neurons.

        R:  np.ndarray
            residuals
    Raises:
        Exception "The number of elements of bl, c1, g, sn must match the number of components"
    """

    logger = logging.getLogger("caiman")
    # tests and initialization
    nr = A.shape[1]
    A = csc_matrix(A)
    if bl is not None and len(bl) != nr:
        raise Exception(
            "The number of elements of bl must match the number of components")
    if c1 is not None and len(c1) != nr:
        raise Exception(
            "The number of elements of c1 must match the number of components")
    if sn is not None and len(sn) != nr:
        raise Exception(
            "The number of elements of sn must match the number of components")
    if g is not None and len(g) != nr:
        raise Exception(
            "The number of elements of g must match the number of components")
    if R is None:
        R = np.zeros_like(C)

    d, t = Y.shape

    # 放在 d, t = Y.shape 之后，所有用到 C/R 之前
    C = np.atleast_2d(C.toarray() if hasattr(C, 'todense') else np.asarray(C)).astype(np.float32)
    R = np.atleast_2d(R.toarray() if hasattr(R, 'todense') else np.asarray(R)).astype(np.float32)
    S = np.atleast_2d(S.toarray() if hasattr(S, 'todense') else np.asarray(S)).astype(np.float32)
    # 现在 C 一定是 (K, T)
    K_actual = C.shape[0]
    # 现在 C、R 一定是 (K, T) 的 ndarray
    # find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.lil_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indices = A_corr[ii, :].nonzero()[1]
        if len(overlap_indices) > 0:
            # we chesk the correlation of the calcium traces for each overlapping components
            C_dense = C.toarray() if hasattr(C, 'todense') else C
            corr_values = [scipy.stats.pearsonr(C[ii], C[jj])[0] for jj in overlap_indices]
            C_corr[ii, overlap_indices] = corr_values

    FF1 = (C_corr + C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(
        FF3)  # % extract connected components

    p = temporal_params['p']
    list_conxcomp_initial = []
    for i in range(nb):  # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp_initial.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp_initial).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((list_conxcomp.shape[1], 1))
        for i in range(np.size(cor)):
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]
        if np.size(cor) > 1:
            # we get the size (indices)
            ind = np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))  # number of merging operations

        if merge_parallel:
            merged_ROIs = [np.where(list_conxcomp[:, ind[i]])[0] for i in range(nbmrg)]
            Acsc_mats = [csc_matrix(A[:, merged_ROI]) for merged_ROI in merged_ROIs]
            Ctmp_mats = [C[merged_ROI] + R[merged_ROI] for merged_ROI in merged_ROIs]
            C_to_norms = [np.sqrt(np.ravel(Acsc.power(2).sum(
                axis=0)) * np.sum(Ctmp ** 2, axis=1)) for (Acsc, Ctmp) in zip(Acsc_mats, Ctmp_mats)]
            indxs = [np.argmax(C_to_norm) for C_to_norm in C_to_norms]
            g_idxs = [merged_ROI[indx] for (merged_ROI, indx) in zip(merged_ROIs, indxs)]
            fms = [fast_merge] * nbmrg
            tps = [temporal_params] * nbmrg
            gs = [g] * nbmrg

            if dview is None:
                merge_res = list(map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            elif 'multiprocessin' in str(type(dview)):
                merge_res = list(
                    dview.map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            else:
                merge_res = list(
                    dview.map_sync(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
                dview.results.clear()
            bl_merged = np.array([res[0] for res in merge_res])
            c1_merged = np.array([res[1] for res in merge_res])
            A_merged = csc_matrix(scipy.sparse.vstack([csc_matrix(res[2]) for res in merge_res]).T)
            C_merged = np.vstack([res[3] for res in merge_res])
            g_merged = np.vstack([res[4] for res in merge_res])
            sn_merged = np.array([res[5] for res in merge_res])
            S_merged = np.vstack([res[6] for res in merge_res])
            R_merged = np.vstack([res[7] for res in merge_res])
        else:
            # we initialize the values
            A_merged = lil_matrix((d, nbmrg))
            C_merged = np.zeros((nbmrg, t))
            R_merged = np.zeros((nbmrg, t))
            S_merged = np.zeros((nbmrg, t))
            bl_merged = np.zeros((nbmrg, 1))
            c1_merged = np.zeros((nbmrg, 1))
            sn_merged = np.zeros((nbmrg, 1))
            g_merged = np.zeros((nbmrg, p))
            merged_ROIs = []
            for i in range(nbmrg):
                merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
                logger.info(f'Merging components {merged_ROI}')
                merged_ROIs.append(merged_ROI)
                Acsc = A.tocsc()[:, merged_ROI]
                Ctmp = C[merged_ROI, :] + R[merged_ROI, :]
                C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                    axis=0)) * np.sum(Ctmp ** 2, axis=1))
                indx = np.argmax(C_to_norm)
                g_idx = [merged_ROI[indx]]
                bm, cm, computedA, computedC, gm, sm, ss, yra = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g,
                                                                                g_idx,
                                                                                indx, temporal_params)

                A_merged[:, i] = csr_matrix(computedA).T
                C_merged[i, :] = computedC
                R_merged[i, :] = yra
                S_merged[i, :] = ss[:t]
                bl_merged[i] = bm
                c1_merged[i] = cm
                sn_merged[i] = sm
                g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            R_merged = R_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        if len(merged_ROIs) > 0:
            # we want to remove merged neuron from the initial part and replace them with merged ones
            neur_id = np.unique(np.hstack(merged_ROIs))
            good_neurons = np.setdiff1d(list(range(nr)), neur_id)
            A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
            C = np.vstack((C[good_neurons, :], C_merged))
            # we continue for the variables
            if S is not None:
                S = np.vstack((S[good_neurons, :], S_merged))
            if R is not None:
                R = np.vstack((R[good_neurons, :], R_merged))
            if bl is not None:
                bl = np.hstack((bl[good_neurons], np.array(bl_merged).flatten()))
            if c1 is not None:
                c1 = np.hstack((c1[good_neurons], np.array(c1_merged).flatten()))
            if sn is not None:
                sn = np.hstack((sn[good_neurons], np.array(sn_merged).flatten()))
            if g is not None:
                g = np.vstack(g)[good_neurons]
                if g.shape[1] == 0:
                    g = np.zeros((len(good_neurons), g_merged.shape[1]))
                g = np.vstack((g, g_merged))

            nr = nr - len(neur_id) + len(C_merged)

    else:
        logger.info('No more components merged!')
        merged_ROIs = []
        empty = []

    return A, C, nr, merged_ROIs, S, bl, c1, sn, g, empty, R


def merge_iter(a):
    Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params = a
    if Ctmp.ndim == 0 or Ctmp.size == 0:
        # 空片段，直接返回空结果
        n = Acsc.shape[0]
        return (np.nan, np.nan, np.zeros(n), np.zeros(0),
                np.zeros(0), np.nan, np.zeros(0), np.zeros(0))
    res = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                          indx, temporal_params)
    return res


def merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params):
    logger = logging.getLogger("caiman")
    if fast_merge:
        # we normalize the values of different A's to be able to compare them efficiently. we then sum them

        computedA = Acsc.dot(C_to_norm)
        for _ in range(10):
            computedC = np.maximum((Acsc.T.dot(computedA)).dot(Ctmp) /
                                   (computedA.T.dot(computedA)), 0)
            nc = computedC.T.dot(computedC)
            if nc == 0:
                break
            computedA = np.maximum(Acsc.dot(Ctmp.dot(computedC.T)) / nc, 0)
    else:
        logger.info('Simple merging ny taking best neuron')
        computedC = Ctmp[indx]
        computedA = Acsc[:, indx]
    # then we de-normalize them using A_to_norm
    A_to_norm = np.sqrt(computedA.T.dot(computedA))  # /Acsc.power(2).sum(0).max())
    computedA /= A_to_norm
    computedC *= A_to_norm
    r = ((Acsc.T.dot(computedA)).dot(Ctmp)) / (computedA.T.dot(computedA)) - computedC
    # we then compute the traces ( deconvolution ) to have a clean c and noise in the background
    c_in = np.array(computedC + r).squeeze()
    if g is not None:
        deconvC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            c_in, g=g_idx, **temporal_params)
    else:
        deconvC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            c_in, g=None, **temporal_params)
    return bm, cm, computedA, deconvC, gm, sm, ss, c_in - deconvC


def pad_est_to_67k(est, T_TARGET=67000):
    """
    把 est 里所有含时间轴的字段循环补齐到 T_TARGET 帧（默认 67 000）
    支持稀疏 / dense / list 三种存储格式
    返回补齐后的 est 对象（原地修改）
    """

    def _do_pad(x, tgt=T_TARGET):
        """真正执行补齐的闭包"""
        if x is None:
            return None
        # ---------- 1. 先转成 ndarray ----------
        if hasattr(x, 'todense'):  # sparse
            arr = x.toarray()
        elif isinstance(x, list):  # list
            arr = np.array(x)
        else:
            arr = np.asarray(x)

        # ---------- 2. 补齐 ----------
        K, T = arr.shape if arr.ndim == 2 else (1, arr.size)
        if T >= tgt:  # 已经够长
            arr = arr[..., :tgt]
        else:  # 需要复制
            n_repeat = int(np.ceil(tgt / T))
            if arr.ndim == 2:
                arr = np.tile(arr, (1, n_repeat))[..., :tgt]
            else:
                arr = np.tile(arr, n_repeat)[:tgt]

        # ---------- 3. 还原成原格式 ----------
        if hasattr(x, 'todense'):  # 还原成 csr
            return csr_matrix(arr)
        elif isinstance(x, list):
            return arr.tolist()
        else:
            return arr

    # 需要补齐的常见字段
    for fld in ('C', 'S', 'R', 'YrA', 'f'):
        if hasattr(est, fld):
            setattr(est, fld, _do_pad(getattr(est, fld)))

    return est


def merge_all_patches(window, overlap_pix, H, W, base_dir, save_dir, prefix_pkl):
    save_dir.mkdir(parents=True, exist_ok=True)
    A_big, C_big, S_big, YrA_big = [], [], [], []
    y_starts = starts_fixed(H, window, overlap_pix)
    x_starts = starts_fixed(W, window, overlap_pix)

    for yi in range(len(y_starts)):
        for xi in range(len(x_starts)):
            pkl_path = base_dir / prefix_pkl.format(yi, xi)
            if not pkl_path.exists():
                print(f'jump {pkl_path}')
                continue
            with open(pkl_path, 'rb') as f:
                est = pickle.load(f)

            y0, x0 = y_starts[yi], x_starts[xi]
            # print('y0:',y0)
            A_pad = pad_a_to_full(est.A, y0, x0, window, H, W)
            # print(f' A_pad.shape={A_pad.shape}')
            A_big.append(A_pad)
            C_big.append(est.C)
            S_big.append(est.S)
            YrA_big.append(est.YrA)
            print(f'append {yi}-{xi} pkl')

    A_big = sparse_hstack(A_big, format='csr')
    C_big = np.vstack(C_big)
    S_big = np.vstack(S_big)
    YrA_big = np.vstack(YrA_big)

    est_big = Estimates()
    est_big.A = A_big
    est_big.C = C_big
    est_big.YrA = YrA_big
    est_big.S = S_big
    est_big.dims = (H, W)
    est_big.bl = []
    est_big.c1 = []
    est_big.sn = []
    est_big.g = []
    out_pkl = save_dir / 'est_big.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(est_big, f)
    print(f'save → {out_pkl}')
    print(f'total num: = {A_big.shape[1]}')


def pad_a_to_full(A, y0, x0, win, H, W):
    K = A.shape[1]
    A_pad = lil_matrix((H * W, K), dtype=A.dtype)
    for k in range(K):
        patch = A[:, k].toarray().reshape(win, win, order='F')

        y_end = min(y0 + win, H)
        x_end = min(x0 + win, W)

        full = np.zeros((H, W), dtype=patch.dtype)
        full[y0:y_end, x0:x_end] = patch
        A_pad[:, k] = full.ravel(order='F')
    return A_pad.tocsr()


def starts_fixed(L, win, ov):
    """返回 0-win-ov-... 像素坐标"""
    st = [0]
    while st[-1] + win - ov <= L - win:
        st.append(st[-1] + win - ov)
    st.append(L - win)  # 确保最后一个窗口始终从边界向内截取
    return st


def to_dense(arr):
    """统一转成 dense ndarray"""
    return np.asarray(arr.todense()) if hasattr(arr, 'todense') else np.asarray(arr)

def stride_cnmf_x(xi, window, H, overlap_pix):
    if window + (xi + 1) * (window - overlap_pix) < H:
        return overlap_pix
    else:
        return window - (H - window - (window - overlap_pix) * xi)

def stride_cnmf_y(yi, window, H, overlap_pix):
    if window + (yi + 1) * (window - overlap_pix) < H:
        return overlap_pix
    else:
        return window - (H - window - (window - overlap_pix) * yi)

# window = 320
# overlap_pix = 20
# H, W = 1670, 1760
# base_dir = Path(r'F:/Comet_patch66')  # 原始 patch 目录
# save_dir = base_dir / 'COMET_patch_merged'

# save_dir.mkdir(parents=True, exist_ok=True)


def merge(num_patches_y, num_patches_x, base_dir, window, overlap_pix, save_dir, H, W):
    os.makedirs(save_dir, exist_ok=True)
    prefix_pkl = 'y{:02d}_x{:02d}_cnm_estimates.pkl'
    middle_dir = 'y{:02d}_x{:02d}/cnmfe'
    # 启动并行
    c, dview, n_proc = cm.cluster.setup_cluster(backend='local', n_processes=None)
    print(f'已启动 {n_proc} 进程')

    # ---------- 1. 逐行合并 ----------
    non_overlap_indices = []
    for yi in range(num_patches_y):
        for xi in range(num_patches_x - 1):
            if xi == 0:  # 第一轮
                file_path1 = Path(base_dir) / middle_dir.format(yi, xi) / prefix_pkl.format(yi, xi)
                file_path2 = Path(base_dir) / middle_dir.format(yi, xi + 1) / prefix_pkl.format(yi, xi + 1)
            else:  # 后续轮
                file_path1 = Path(save_dir) / prefix_pkl.format(yi, xi)
                file_path2 = Path(base_dir) / middle_dir.format(yi, xi + 1) / prefix_pkl.format(yi, xi + 1)  # 右侧块还没被写过

            if not os.path.exists(file_path1):
                raise FileNotFoundError(f'File not found: {file_path1}')
            if not os.path.exists(file_path2):
                raise FileNotFoundError(f'File not found: {file_path2}')
            with open(file_path1, 'rb') as f:
                est1 = pickle.load(f)
            with open(file_path2, 'rb') as f:
                est2 = pickle.load(f)
            K_i1 = est1.A.shape[1]
            K_i2 = est2.A.shape[1]
            dim = np.sqrt(est1.A.shape[0])
            A_list = []
            C_list = []
            S_list = []
            R_list = []
            bl_list = []
            c1_list = []
            sn_list = []
            g_list = []
            overlap_fp = []
            overlap_C = []
            overlap_S = []
            overlap_bl = []
            overlap_c1 = []
            overlap_g = []
            overlap_YrA = []
            overlap_sn = []
            del_fp_1 = []
            del_C_1 = []
            del_S_1 = []
            del_b1 = []
            del_c1 = []
            del_g1 = []
            del_YrA1 = []
            del_sn1 = []
            del_fp_2 = []
            del_C_2 = []
            del_S_2 = []
            del_b2 = []
            del_c2 = []
            del_g2 = []
            del_YrA2 = []
            del_sn2 = []

            # 左侧
            non_overlap_indices1 = []
            for k in range(K_i1):
                fp_2d = est1.A[:, k].toarray().reshape(window, window, order='F')
                stridex = stride_cnmf_x(xi=xi, window=window, H=W, overlap_pix=overlap_pix)
                fp_2d_overlap = fp_2d[:, -stridex:]
                y = np.arange(dim)
                x = np.arange(dim)
                X, Y = np.meshgrid(x, y)
                if fp_2d_overlap.sum() > 0 and fp_2d_overlap.sum() > 0.8 * fp_2d.sum():
                    overlap_fp.append(fp_2d_overlap.ravel(order='F'))
                    # print('Cshape',est1.C[k, :].shape)
                    overlap_C.append(est1.C[k, :].reshape(1, -1))
                    # print(overlap_C.shape)
                    overlap_S.append(est1.S[k, :].reshape(1, -1))
                    overlap_YrA.append(est1.YrA[k, :].reshape(1, -1))
                    # overlap_c1.append(est1.c1[k])
                    # overlap_bl.append(est1.bl[k])
                    # overlap_g.append(est1.g[k])
                    # overlap_sn.append(est1.sn[k])
                else:
                    non_overlap_indices1.append(k)
                    # print('non_overlap_id',non_overlap_indices1)

            j_nonoverlap = 0
            for j in range(K_i1):
                if j_nonoverlap >= len(non_overlap_indices1):
                    break
                if j == non_overlap_indices1[j_nonoverlap]:
                    j_nonoverlap = j_nonoverlap + 1
                    del_fp_1.append(est1.A[:, j].toarray().reshape(1, -1))
                    del_C_1.append(est1.C[j, :].reshape(1, -1))
                    del_S_1.append(est1.S[j, :].reshape(1, -1))
                    # del_b1.append(est1.bl[j])
                    # del_c1.append(est1.c1[j])
                    # del_g1.append(est1.g[j])
                    del_YrA1.append(est1.YrA[j, :].reshape(1, -1))
                    # del_sn1.append(est1.sn[j])
                else:
                    continue

            print('overlap_num-1', len(overlap_fp))
            del_fp_1 = np.vstack(del_fp_1).T
            del_C_1 = np.vstack(del_C_1)
            del_S_1 = np.vstack(del_S_1)
            del_YrA1 = np.vstack(del_YrA1)

            # 右侧
            non_overlap_indices2 = []
            for k in range(K_i2):
                fp_2d = est2.A[:, k].toarray().reshape(window, window, order='F')
                fp_2d_overlap = fp_2d[:, :stride_cnmf_x(xi=xi, window=window, H=W, overlap_pix=overlap_pix)]
                y = np.arange(dim)
                x = np.arange(dim)
                X, Y = np.meshgrid(x, y)
                if fp_2d_overlap.sum() > 0 and fp_2d_overlap.sum() > 0.8 * fp_2d.sum():
                    overlap_fp.append(fp_2d_overlap.ravel(order='F'))
                    overlap_C.append(est2.C[k, :].reshape(1, -1))
                    overlap_S.append(est2.S[k, :].reshape(1, -1))
                    overlap_YrA.append(est2.YrA[k, :].reshape(1, -1))
                    # overlap_c1.append(est2.c1[k])
                    # overlap_bl.append(est2.bl[k])
                    # overlap_g.append(est2.g[k])
                    # overlap_sn.append(est2.sn[k])
                else:
                    non_overlap_indices2.append(k)

            j_nonoverlap = 0
            for j in range(K_i2):
                if j_nonoverlap >= len(non_overlap_indices2):
                    break
                if j == non_overlap_indices2[j_nonoverlap]:
                    j_nonoverlap = j_nonoverlap + 1
                    del_fp_2.append(est2.A[:, j].toarray().reshape(1, -1))
                    del_C_2.append(est2.C[j, :].reshape(1, -1))
                    del_S_2.append(est2.S[j, :].reshape(1, -1))
                    # del_b2.append(est2.bl[j])
                    # del_c2.append(est2.c1[j])
                    # del_g2.append(est2.g[j])
                    del_YrA2.append(est2.YrA[j, :].reshape(1, -1))
                    # del_sn2.append(est2.sn[j])
                else:
                    continue
            del_fp_2 = np.vstack(del_fp_2).T
            del_C_2 = np.vstack(del_C_2)
            del_S_2 = np.vstack(del_S_2)
            del_YrA2 = np.vstack(del_YrA2)
            print('右侧块初始神经元:', est2.A.shape[1])
            print('右侧块剩余神经元:', del_fp_2.shape)

            if not overlap_fp:  # 空列表
                print(f'补丁对 ({yi},{xi})-({yi},{xi + 1}) 无重叠神经元，跳过合并')
                with open(Path(save_dir) / prefix_pkl.format(yi, xi), 'wb') as f:
                    pickle.dump(est1, f)
                print(f'已保存 {file_path1}')
                with open(Path(save_dir) / prefix_pkl.format(yi, xi + 1), 'wb') as f:
                    pickle.dump(est2, f)
                print(f'已保存 {file_path2}')
                continue  # 直接处理下一对
            else:
                overlap_fp = np.vstack(overlap_fp).T
                A_list = (csr_matrix(overlap_fp))

            # ----------------------------看上面部分---------------------------------------

            # 其余矩阵转 dense
            C_list = np.vstack(overlap_C)
            S_list = np.vstack(overlap_S)
            R_list = np.vstack(overlap_YrA)
            # bl_list.append(np.array(overlap_bl))
            # c1_list.append(np.array(overlap_c1))
            # sn_list.append(np.asarray(overlap_sn))
            # g_list.append(np.array(overlap_g))
            print('Clist_shape', C_list.shape)

            # ---------- 2. 拼接overlap列表 ----------
            A_big = A_list
            C_big = C_list
            S_big = S_list
            R_big = R_list
            # bl_big = np.concatenate(bl_list)
            # c1_big = np.concatenate(c1_list)
            # sn_big = np.concatenate(sn_list)
            # g_big = np.concatenate(g_list, axis=0)

            T = C_big.shape[1]
            Y_big = np.zeros((window * stride_cnmf_x(xi=xi, window=window, H=W, overlap_pix=overlap_pix), T),
                             dtype=np.float32)
            f_big = np.zeros(T)
            print("A.shape:", A_big.shape)
            # print("C.shape:", C_big.shape if hasattr(C_big, 'shape') else np.array(C_big).shape)
            print("C_big.shape =", C_big.shape)
            print("R_big.shape =", R_big.shape)
            # print('bl.shape =',bl_big.shape)
            # print('bl_1.shape =',np.array(del_b1).shape)
            # print('bl_2.shape =',np.array(del_b2).shape)
            # ---------- 3. 合并 ----------
            A_new, C_new, K_new, merged_ROIs, S_new, bl_new, c1_new, sn_new, g_new, empty, R_new = \
                merge_components(Y=Y_big,
                                 A=A_big,
                                 b=np.zeros(Y_big.shape[0]),  # 或者你之前估计的 b
                                 C=C_big,
                                 R=R_big,
                                 f=np.zeros(T),  # 或者你之前估计的 f
                                 S=S_big,
                                 sn_pix=np.ones(Y_big.shape[0]),
                                 temporal_params={'p': 1},
                                 spatial_params={},
                                 dview=dview,
                                 thr=0.85,
                                 fast_merge=True)
            print('A_newshape', A_new.shape)
            print('del_fp_1', del_fp_1.shape)
            # print('bl_new.shape =',np.array(bl_new).shape)
            K_num = A_new.shape[1]
            full_fp = np.zeros((window * window, K_num), dtype=np.float32)
            for ll in range(K_num):
                fp_2d1 = A_new[:, ll].toarray().reshape(window, stride_cnmf_x(xi=xi, window=window, H=W,
                                                                              overlap_pix=overlap_pix), order='F')
                full_2d = np.zeros((window, window), dtype=np.float32)
                full_2d[:, -stride_cnmf_x(xi=xi, window=window, H=W, overlap_pix=overlap_pix):] = fp_2d1
                full_fp[:, ll] = full_2d.ravel(order='F')
            print('A_newpaddingshape', full_fp.shape)

            # del_C_1.append(C_new)
            # del_S_1.append(S_new)
            # del_b1.append(bl_new)
            # del_c1.append(c1_new)
            # del_g1.append(g_new)
            # del_YrA1.append(R_new)
            # del_sn1.append(sn_new)
            # bl_new = np.atleast_1d(bl_new)
            # c1_new = np.atleast_1d(c1_new)
            # sn_new = np.atleast_1d(sn_new)
            # g_new = np.atleast_1d(g_new)
            A_1 = np.hstack((del_fp_1, full_fp))
            C_1 = np.vstack((del_C_1, C_new))
            S_1 = np.vstack((del_S_1, S_new))
            R_1 = np.vstack((del_YrA1, R_new))
            # bl_1 = np.concatenate((del_b1,bl_new))
            # c1_1 = np.concatenate((del_c1,c1_new))
            # sn_1 = np.concatenate((del_sn1,sn_new))
            # g_1 = np.concatenate((np.array(del_g1).ravel(), g_new), axis=0)
            print('A_1shape', A_1.shape)
            print('C_1shape', C_1.shape)

            # ---------- 4. 保存行 ----------
            # 把 dict 换成对象（最简单做法：原 est + 更新字段）
            # 假设 est1 是上一轮原始 CNMF 结果
            est1.A = csr_matrix(A_1)  # 更新空间分量
            est1.C = C_1  # 更新时间分量
            est1.S = S_1
            est1.R = R_1
            est1.YrA = R_1
            '''
            est1.bl = bl_1
            est1.c1 = c1_1
            est1.sn = sn_1
            est1.g  = g_1
            '''
            est1.bl = []
            est1.c1 = []
            est1.sn = []
            est1.g = []
            est1.dims = (window, window)

            with open(Path(save_dir) / prefix_pkl.format(yi, xi), 'wb') as f:
                pickle.dump(est1, f)

            print(f'已保存 {file_path1}')

            est2.A = csr_matrix(del_fp_2)
            est2.C = del_C_2
            est2.S = del_S_2
            est2.R = del_YrA2
            est2.YrA = del_YrA2
            '''
            est2.bl = del_b2
            est2.c1 = del_c2
            est2.sn = del_sn2
            est2.g  = del_g2
            est2.dims = (window, window)
            '''
            est2.bl = []
            est2.c1 = []
            est2.sn = []
            est2.g = []
            with open(Path(save_dir) / prefix_pkl.format(yi, xi + 1), 'wb') as f:
                pickle.dump(est2, f)
            print(f'已保存 {file_path2}')

    # ---------- 1. 逐列合并 ----------
    non_overlap_indices = []
    for xi in range(num_patches_x):
        for yi in range(num_patches_y - 1):
            file_path1 = Path(save_dir) / prefix_pkl.format(yi, xi)  # 读行合并后的
            file_path2 = Path(save_dir) / prefix_pkl.format(yi + 1, xi)
            if not os.path.exists(file_path1):
                raise FileNotFoundError(f'File not found: {file_path1}')
            if not os.path.exists(file_path2):
                raise FileNotFoundError(f'File not found: {file_path2}')
            with open(file_path1, 'rb') as f:
                est1 = pickle.load(f)
            # est1 = pad_est_to_67k(est1)
            with open(file_path2, 'rb') as f:
                est2 = pickle.load(f)
            # est2 = pad_est_to_67k(est2)
            K_i1 = est1.A.shape[1]
            K_i2 = est2.A.shape[1]
            dim = np.sqrt(est1.A.shape[0])
            A_list = []
            C_list = []
            S_list = []
            R_list = []
            bl_list = []
            c1_list = []
            sn_list = []
            g_list = []
            overlap_fp = []
            overlap_C = []
            overlap_S = []
            overlap_bl = []
            overlap_c1 = []
            overlap_g = []
            overlap_YrA = []
            overlap_sn = []
            del_fp_1 = []
            del_C_1 = []
            del_S_1 = []
            del_b1 = []
            del_c1 = []
            del_g1 = []
            del_YrA1 = []
            del_sn1 = []
            del_fp_2 = []
            del_C_2 = []
            del_S_2 = []
            del_b2 = []
            del_c2 = []
            del_g2 = []
            del_YrA2 = []
            del_sn2 = []

            # 上侧
            non_overlap_indices1 = []
            for k in range(K_i1):
                fp_2d = est1.A[:, k].toarray().reshape(window, window, order='F')
                stridex = stride_cnmf_y(yi=yi, window=window, H=H, overlap_pix=overlap_pix)
                fp_2d_overlap = fp_2d[-stridex:, :]
                y = np.arange(dim)
                x = np.arange(dim)
                X, Y = np.meshgrid(x, y)
                if fp_2d_overlap.sum() > 0 and fp_2d_overlap.sum() > 0.8 * fp_2d.sum():
                    overlap_fp.append(fp_2d_overlap.ravel(order='F'))
                    # print('Cshape',est1.C[k, :].shape)
                    overlap_C.append(est1.C[k, :].reshape(1, -1))
                    # print(overlap_C.shape)
                    overlap_S.append(est1.S[k, :].reshape(1, -1))
                    overlap_YrA.append(est1.YrA[k, :].reshape(1, -1))
                    # overlap_c1.append(est1.c1[k])
                    # overlap_bl.append(est1.bl[k])
                    # overlap_g.append(est1.g[k])
                    # overlap_sn.append(est1.sn[k])
                else:
                    non_overlap_indices1.append(k)
                    # print('non_overlap_id',non_overlap_indices1)

            j_nonoverlap = 0
            for j in range(K_i1):
                if j_nonoverlap >= len(non_overlap_indices1):
                    break
                if j == non_overlap_indices1[j_nonoverlap]:
                    j_nonoverlap = j_nonoverlap + 1
                    del_fp_1.append(est1.A[:, j].toarray().reshape(1, -1))
                    del_C_1.append(est1.C[j, :].reshape(1, -1))
                    del_S_1.append(est1.S[j, :].reshape(1, -1))
                    # del_b1.append(est1.bl[j])
                    # del_c1.append(est1.c1[j])
                    # del_g1.append(est1.g[j])
                    del_YrA1.append(est1.YrA[j, :].reshape(1, -1))
                    # del_sn1.append(est1.sn[j])
                else:
                    continue

            print('overlap_num-1', len(overlap_fp))
            del_fp_1 = np.vstack(del_fp_1).T
            del_C_1 = np.vstack(del_C_1)
            del_S_1 = np.vstack(del_S_1)
            del_YrA1 = np.vstack(del_YrA1)

            # 下侧
            non_overlap_indices2 = []
            for k in range(K_i2):
                fp_2d = est2.A[:, k].toarray().reshape(window, window, order='F')
                fp_2d_overlap = fp_2d[:stride_cnmf_y(yi=yi, window=window, H=H, overlap_pix=overlap_pix), :]
                y = np.arange(dim)
                x = np.arange(dim)
                X, Y = np.meshgrid(x, y)
                if fp_2d_overlap.sum() > 0 and fp_2d_overlap.sum() > 0.8 * fp_2d.sum():
                    overlap_fp.append(fp_2d_overlap.ravel(order='F'))
                    overlap_C.append(est2.C[k, :].reshape(1, -1))
                    overlap_S.append(est2.S[k, :].reshape(1, -1))
                    overlap_YrA.append(est2.YrA[k, :].reshape(1, -1))
                    # overlap_c1.append(est2.c1[k])
                    # overlap_bl.append(est2.bl[k])
                    # overlap_g.append(est2.g[k])
                    # overlap_sn.append(est2.sn[k])
                else:
                    non_overlap_indices2.append(k)

            j_nonoverlap = 0
            for j in range(K_i2):
                if j_nonoverlap >= len(non_overlap_indices2):
                    break
                if j == non_overlap_indices2[j_nonoverlap]:
                    j_nonoverlap = j_nonoverlap + 1
                    del_fp_2.append(est2.A[:, j].toarray().reshape(1, -1))
                    del_C_2.append(est2.C[j, :].reshape(1, -1))
                    del_S_2.append(est2.S[j, :].reshape(1, -1))
                    # del_b2.append(est2.bl[j])
                    # del_c2.append(est2.c1[j])
                    # del_g2.append(est2.g[j])
                    del_YrA2.append(est2.YrA[j, :].reshape(1, -1))
                    # del_sn2.append(est2.sn[j])
                else:
                    continue
            del_fp_2 = np.vstack(del_fp_2).T
            del_C_2 = np.vstack(del_C_2)
            del_S_2 = np.vstack(del_S_2)
            del_YrA2 = np.vstack(del_YrA2)
            if not overlap_fp:  # 空列表
                print(f'补丁对 ({yi},{xi})-({yi + 1},{xi}) 无重叠神经元，跳过合并')
                continue  # 直接处理下一对
            else:
                overlap_fp = np.vstack(overlap_fp).T
                # print(overlap_fp.shape)
            A_list = (csr_matrix(overlap_fp))

            # ----------------------------看上面部分---------------------------------------

            # 其余矩阵转 dense
            C_list = np.vstack(overlap_C)
            S_list = np.vstack(overlap_S)
            R_list = np.vstack(overlap_YrA)
            # bl_list.append(np.array(overlap_bl))
            # c1_list.append(np.array(overlap_c1))
            # sn_list.append(np.asarray(overlap_sn))
            # g_list.append(np.array(overlap_g))
            print('Clist_shape', C_list.shape)

            # ---------- 2. 拼接overlap列表 ----------
            A_big = A_list
            C_big = C_list
            S_big = S_list
            R_big = R_list
            # bl_big = np.concatenate(bl_list)
            # c1_big = np.concatenate(c1_list)
            # sn_big = np.concatenate(sn_list)
            # g_big = np.concatenate(g_list, axis=0)

            T = C_big.shape[1]
            Y_big = np.zeros((window * stride_cnmf_y(yi=yi, window=window, H=H, overlap_pix=overlap_pix), T),
                             dtype=np.float32)
            f_big = np.zeros(T)
            print("A.shape:", A_big.shape)
            print("C.shape:", C_big.shape if hasattr(C_big, 'shape') else np.array(C_big).shape)
            # print("C_big.shape =", C_big.shape)
            print("R_big.shape =", R_big.shape)
            # print('bl.shape =',bl_big.shape)
            # ---------- 3. 合并 ----------
            A_new, C_new, K_new, merged_ROIs, S_new, bl_new, c1_new, sn_new, g_new, empty, R_new = \
                merge_components(Y=Y_big,
                                 A=A_big,
                                 b=np.zeros(Y_big.shape[0]),  # 或者你之前估计的 b
                                 C=C_big,
                                 R=R_big,
                                 f=np.zeros(T),  # 或者你之前估计的 f
                                 S=S_big,
                                 sn_pix=np.ones(Y_big.shape[0]),
                                 temporal_params={'p': 1},
                                 spatial_params={},
                                 dview=dview,
                                 thr=0.85,
                                 fast_merge=True)
            print('A_newshape', A_new.shape)
            print('del_fp_1', del_fp_1.shape)

            K_num = A_new.shape[1]
            full_fp = np.zeros((window * window, K_num), dtype=np.float32)
            for ll in range(K_num):
                fp_2d1 = A_new[:, ll].toarray().reshape(
                    stride_cnmf_y(yi=yi, window=window, H=H, overlap_pix=overlap_pix), window, order='F')
                full_2d = np.zeros((window, window), dtype=np.float32)
                full_2d[-stride_cnmf_y(yi=yi, window=window, H=H, overlap_pix=overlap_pix):, :] = fp_2d1
                full_fp[:, ll] = full_2d.ravel(order='F')
            print('A_newpaddingshape', full_fp.shape)

            # del_C_1.append(C_new)
            # del_S_1.append(S_new)
            # del_b1.append(bl_new)
            # del_c1.append(c1_new)
            # del_g1.append(g_new)
            # del_YrA1.append(R_new)
            # del_sn1.append(sn_new)
            bl_new = np.atleast_1d(bl_new)
            c1_new = np.atleast_1d(c1_new)
            sn_new = np.atleast_1d(sn_new)
            g_new = np.atleast_1d(g_new)
            A_1 = np.hstack((del_fp_1, full_fp))
            C_1 = np.vstack((del_C_1, C_new))
            S_1 = np.vstack((del_S_1, S_new))
            R_1 = np.vstack((del_YrA1, R_new))
            # bl_1 = np.concatenate((del_b1,bl_new))
            # c1_1 = np.concatenate((del_c1,c1_new))
            # sn_1 = np.concatenate((del_sn1,sn_new))
            # g_1 = np.concatenate((np.array(del_g1).ravel(), g_new), axis=0)
            print('A_1shape', A_1.shape)
            print('C_1shape', C_1.shape)

            # ---------- 4. 保存行 ----------
            # 把 dict 换成对象（最简单做法：原 est + 更新字段）
            # 假设 est1 是上一轮原始 CNMF 结果
            est1.A = csr_matrix(A_1)  # 更新空间分量
            est1.C = C_1  # 更新时间分量
            est1.S = S_1
            est1.R = R_1
            est1.YrA = R_1
            est1.bl = []
            est1.c1 = []
            est1.sn = []
            est1.g = []
            est1.dims = (window, window)

            with open(Path(save_dir) / prefix_pkl.format(yi, xi), 'wb') as f:
                pickle.dump(est1, f)

            print(f'已覆盖 {file_path1}')

            est2.A = csr_matrix(del_fp_2)
            est2.C = del_C_2
            est2.S = del_S_2
            est2.R = del_YrA2
            est2.YrA = del_YrA2
            est2.bl = []
            est2.c1 = []
            est2.sn = []
            est2.g = []
            est2.dims = (window, window)

            with open(Path(save_dir) / prefix_pkl.format(yi + 1, xi), 'wb') as f:
                pickle.dump(est2, f)

            print(f'已覆盖 {file_path2}')
    print('done!')
    cm.stop_server(dview=dview)
    merge_all_patches(window=window, overlap_pix=overlap_pix, H=H, W=W, base_dir=Path(save_dir),
                      save_dir=Path(save_dir) / 'COMET_patch_merged_all', prefix_pkl=prefix_pkl)
