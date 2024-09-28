import numpy as np
from scipy.signal import convolve2d
from Qabf import get_Qabf
from Nabf import get_Nabf
import math
import torch
import torch.nn.functional as F
from ssim import ssim, ms_ssim

def EN_function(image_tensor):
    # 计算直方图
    histogram = torch.histc(image_tensor, bins=256, min=0, max=255)
    # 归一化直方图
    histogram = histogram / histogram.sum()
    # 计算熵
    entropy = -torch.sum(histogram * torch.log2(histogram + 1e-7))
    return entropy


def SF_function(image_tensor):

    # 计算行差分和列差分
    RF = image_tensor[1:, :] - image_tensor[:-1, :]
    CF = image_tensor[:, 1:] - image_tensor[:, :-1]

    # 计算均值平方根
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF1 = torch.sqrt(torch.mean(CF ** 2))

    # 计算 SF
    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF


def SD_function(image_tensor):
    m, n = image_tensor.shape
    u = torch.mean(image_tensor)  # 计算均值
    SD = torch.sqrt(torch.sum((image_tensor - u) ** 2) / (m * n))  # 计算标准差
    return SD

def PSNR_function(A, B, F):
    # 确保输入是浮点数张量并在 GPU 上
    A = A.float() / 255.0
    B = B.float() / 255.0
    F = F.float() / 255.0

    m, n = F.shape
    MSE_AF = torch.mean((F - A) ** 2)
    MSE_BF = torch.mean((F - B) ** 2)

    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * torch.log10(255 / torch.sqrt(MSE))

    return PSNR


def MSE_function(A, B, F):
    # 确保输入是浮点数张量并在 GPU 上
    A = A.float() / 255.0
    B = B.float() / 255.0
    F = F.float() / 255.0

    m, n = F.shape
    MSE_AF = torch.mean((F - A) ** 2)
    MSE_BF = torch.mean((F - B) ** 2)

    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE

def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

import torch
import torch.nn.functional as F

def fspecial_gaussian(size, sigma):
    """Create a Gaussian filter."""
    x = torch.linspace(-size[0]//2, size[0]//2, size[0])
    y = torch.linspace(-size[1]//2, size[1]//2, size[1])
    x, y = torch.meshgrid(x, y)
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d(input, kernel):
    """Perform 2D convolution using PyTorch."""
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(input.device)  # Add batch and channel dimensions
    return F.conv2d(input.unsqueeze(0).unsqueeze(0), kernel, padding=kernel.shape[2] // 2)[0][0]

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win)
            dist = convolve2d(dist, win)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win)
        mu2 = convolve2d(dist, win)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win) - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win) - mu2_sq
        sigma12 = convolve2d(ref * dist, win) - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += torch.sum(torch.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp

def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


def CC_function(A, B, F):
    rAF = torch.sum((A - torch.mean(A)) * (F - torch.mean(F))) / torch.sqrt(torch.sum((A - torch.mean(A)) ** 2) * torch.sum((F - torch.mean(F)) ** 2))
    rBF = torch.sum((B - torch.mean(B)) * (F - torch.mean(F))) / torch.sqrt(torch.sum((B - torch.mean(B)) ** 2) * torch.sum((F - torch.mean(F)) ** 2))
    CC = torch.mean(torch.tensor([rAF, rBF]))
    return CC

def corr2(a, b):
    a = a - torch.mean(a)
    b = b - torch.mean(b)
    r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    return r

def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r

def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)

def Nabf_function(A, B, F):
    return Nabf_function(A, B, F)


def Hab(im1, im2, gray_level):
	hang, lie = im1.shape
	count = hang * lie
	N = gray_level
	h = np.zeros((N, N))
	for i in range(hang):
		for j in range(lie):
			h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
	h = h / np.sum(h)
	im1_marg = np.sum(h, axis=0)
	im2_marg = np.sum(h, axis=1)
	H_x = 0
	H_y = 0
	for i in range(N):
		if (im1_marg[i] != 0):
			H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
	for i in range(N):
		if (im2_marg[i] != 0):
			H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
	H_xy = 0
	for i in range(N):
		for j in range(N):
			if (h[i, j] != 0):
				H_xy = H_xy + h[i, j] * math.log2(h[i, j])
	MI = H_xy - H_x - H_y
	return MI

def MI_function(A, B, F, gray_level=256):
	MIA = Hab(A, F, gray_level)
	MIB = Hab(B, F, gray_level)
	MI_results = MIA + MIB
	return MI_results


def AG_function(image_tensor):
    # 计算梯度
    grady, gradx = torch.gradient(image_tensor)
    # 计算梯度的平方和
    s = torch.sqrt((gradx ** 2 + grady ** 2) / 2)

    # 计算平均梯度
    AG = torch.sum(s) / (image_tensor.shape[0] * image_tensor.shape[1])
    return AG

def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()

def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = 1 * ssim_A + 1 * ssim_B
    return MS_SSIM.item()

def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf
