import numpy as np
from scipy.signal import convolve2d
from Qabf import get_Qabf
from Nabf import get_Nabf
import math
import torch
import torch.nn.functional as F
import torch.fft
from ssim import ssim, ms_ssim
from sklearn.metrics import normalized_mutual_info_score

def EN_function(image_tensor):
    histogram = torch.histc(image_tensor, bins=256, min=0, max=255)
    histogram = histogram / histogram.sum()
    entropy = -torch.sum(histogram * torch.log2(histogram + 1e-7))
    return entropy

def CE_function(ir_img_tensor, vi_img_tensor, f_img_tensor):
    ir_img_tensor = torch.sigmoid(ir_img_tensor)
    vi_img_tensor = torch.sigmoid(vi_img_tensor)
    f_img_tensor = torch.sigmoid(f_img_tensor)
    epsilon = 1e-7
    f_img_tensor = torch.clamp(f_img_tensor, epsilon, 1.0 - epsilon)
    true_tensor = (ir_img_tensor + vi_img_tensor) / 2
    true_tensor = torch.clamp(true_tensor, epsilon, 1.0 - epsilon)
    CE = F.binary_cross_entropy(f_img_tensor, true_tensor)
    return CE

def QNCIE_function(ir_img_tensor, vi_img_tensor, f_img_tensor):
    def normalize1(img_tensor):
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        return (img_tensor - img_min) / (img_max - img_min)
    def NCC(img1, img2):
        mean1 = torch.mean(img1)
        mean2 = torch.mean(img2)
        numerator = torch.sum((img1 - mean1) * (img2 - mean2))
        denominator = torch.sqrt(torch.sum((img1 - mean1) ** 2) * torch.sum((img2 - mean2) ** 2))
        return numerator / (denominator + 1e-10)

    ir_img_tensor = normalize1(ir_img_tensor)
    vi_img_tensor = normalize1(vi_img_tensor)
    f_img_tensor = normalize1(f_img_tensor)

    NCCxy = NCC(ir_img_tensor, vi_img_tensor)
    NCCxf = NCC(ir_img_tensor, f_img_tensor)
    NCCyf = NCC(vi_img_tensor, f_img_tensor)
    R = torch.tensor([[1, NCCxy, NCCxf],
                      [NCCxy, 1, NCCyf],
                      [NCCxf, NCCyf, 1]], dtype=torch.float32)

    r = torch.linalg.eigvals(R).real
    K = 3
    b = 256
    HR = torch.sum(r * torch.log2(r / K) / K)
    HR = -HR / np.log2(b)
    QNCIE = 1 - HR.item()
    return QNCIE


def TE_function(ir_img_tensor, vi_img_tensor, f_img_tensor, q=1, ksize=256):
    def compute_entropy(img_tensor, q, ksize):
        img_tensor = img_tensor.view(-1).float()
        histogram = torch.histc(img_tensor, bins=ksize, min=0, max=ksize - 1)
        probabilities = histogram / torch.sum(histogram)
        if q == 1:
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
        else:
            entropy = (1 / (q - 1)) * (1 - torch.sum(probabilities ** q))
        return entropy.item()

    TE_ir = compute_entropy(ir_img_tensor, q, ksize)
    TE_vi = compute_entropy(vi_img_tensor, q, ksize)
    TE_f = compute_entropy(f_img_tensor, q, ksize)
    TE = TE_ir + TE_vi - TE_f
    return TE


def EI_function(f_img_tensor):
    sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]]).to(f_img_tensor.device)

    sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]]).to(f_img_tensor.device)

    sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)
    gx = F.conv2d(f_img_tensor.unsqueeze(0).unsqueeze(0), sobel_kernel_x, padding=1)
    gy = F.conv2d(f_img_tensor.unsqueeze(0).unsqueeze(0), sobel_kernel_y, padding=1)

    g = torch.sqrt(gx ** 2 + gy ** 2)
    EI = torch.mean(g).item()

    return EI


def SF_function(image_tensor):

    RF = image_tensor[1:, :] - image_tensor[:-1, :]
    CF = image_tensor[:, 1:] - image_tensor[:, :-1]
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF1 = torch.sqrt(torch.mean(CF ** 2))

    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF


def SD_function(image_tensor):
    m, n = image_tensor.shape
    u = torch.mean(image_tensor)
    SD = torch.sqrt(torch.sum((image_tensor - u) ** 2) / (m * n))
    return SD

def PSNR_function(A, B, F):
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
    A = A.float() / 255.0
    B = B.float() / 255.0
    F = F.float() / 255.0

    m, n = F.shape
    MSE_AF = torch.mean((F - A) ** 2)
    MSE_BF = torch.mean((F - B) ** 2)

    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE

def fspecial_gaussian(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def fspecial_gaussian(size, sigma):
    x = torch.linspace(-size[0]//2, size[0]//2, size[0])
    y = torch.linspace(-size[1]//2, size[1]//2, size[1])
    x, y = torch.meshgrid(x, y)
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d(input, kernel):
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

def entropy(im, gray_level=256):

    hist, _ = np.histogram(im, bins=gray_level, range=(0, gray_level), density=True)
    H = -np.sum(hist * np.log2(hist + 1e-10))
    return H

def NMI_function(A, B, F, gray_level=256):

    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI_results = MIA + MIB

    H_A = entropy(A, gray_level)
    H_B = entropy(B, gray_level)

    NMI = 2 * MI_results / (H_A + H_B + 1e-10)

    return NMI

def AG_function(image_tensor):
    grady, gradx = torch.gradient(image_tensor)
    s = torch.sqrt((gradx ** 2 + grady ** 2) / 2)
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

def Qy_function(ir_img_tensor, vi_img_tensor, f_img_tensor):
    def gaussian_filter(window_size, sigma):
        gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)], device=ir_img_tensor.device)
        gauss = gauss / gauss.sum()
        gauss = gauss.view(1, 1, -1).repeat(1, 1, 1, 1)
        return gauss

    def ssim_yang(img1, img2):
        window_size = 7
        sigma = 1.5
        window = gaussian_filter(window_size, sigma)
        window = window.expand(1, 1, window_size, window_size)

        mu1 = F.conv2d(img1, window, stride=1, padding=window_size // 2)
        mu2 = F.conv2d(img2, window, stride=1, padding=window_size // 2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1.pow(2), window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2.pow(2), window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        mssim = ssim_map.mean().item()

        return mssim, ssim_map, sigma1_sq, sigma2_sq

    ir_img_tensor = ir_img_tensor.unsqueeze(0).unsqueeze(0).double()
    vi_img_tensor = vi_img_tensor.unsqueeze(0).unsqueeze(0).double()
    f_img_tensor = f_img_tensor.unsqueeze(0).unsqueeze(0).double()

    _, ssim_map1, sigma1_sq1, sigma2_sq1 = ssim_yang(ir_img_tensor, vi_img_tensor)
    _, ssim_map2, _, _ = ssim_yang(ir_img_tensor, f_img_tensor)
    _, ssim_map3, _, _ = ssim_yang(vi_img_tensor, f_img_tensor)
    bin_map = (ssim_map1 >= 0.75).double()
    ramda = sigma1_sq1 / (sigma1_sq1 + sigma2_sq1 + 1e-10)

    Q1 = (ramda * ssim_map2 + (1 - ramda) * ssim_map3) * bin_map
    Q2 = torch.max(ssim_map2, ssim_map3) * (1 - bin_map)
    Qy = (Q1 + Q2).mean().item()

    return Qy

def gaussian2d(n1, n2, sigma, device):
    x = torch.arange(-15, 16, device=device, dtype=torch.double)
    y = torch.arange(-15, 16, device=device, dtype=torch.double)
    x, y = torch.meshgrid(x, y)
    G = torch.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * torch.pi * sigma**2)
    return G

def contrast(G1, G2, img):
    buff = F.conv2d(img.unsqueeze(0).unsqueeze(0), G1.unsqueeze(0).unsqueeze(0), padding=G1.shape[-1] // 2)
    buff1 = F.conv2d(img.unsqueeze(0).unsqueeze(0), G2.unsqueeze(0).unsqueeze(0), padding=G2.shape[-1] // 2)
    return buff / (buff1 + 1e-10) - 1

def Qcb_function(ir_img_tensor, vi_img_tensor, f_img_tensor):
    device = ir_img_tensor.device

    ir_img_tensor = ir_img_tensor.double().to(device)
    vi_img_tensor = vi_img_tensor.double().to(device)
    f_img_tensor = f_img_tensor.double().to(device)
    ir_img_tensor = (ir_img_tensor - ir_img_tensor.min()) / (ir_img_tensor.max() - ir_img_tensor.min())
    vi_img_tensor = (vi_img_tensor - vi_img_tensor.min()) / (vi_img_tensor.max() - vi_img_tensor.min())
    f_img_tensor = (f_img_tensor - f_img_tensor.min()) / (f_img_tensor.max() - f_img_tensor.min())

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622
    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001
    hang, lie = ir_img_tensor.shape[-2:]

    u, v = torch.meshgrid(torch.fft.fftfreq(hang, device=device), torch.fft.fftfreq(lie, device=device), indexing='ij')
    u = u * (hang / 30)
    v = v * (lie / 30)
    r = torch.sqrt(u**2 + v**2)
    Sd = (torch.exp(-(r / f0)**2) - a * torch.exp(-(r / f1)**2)).to(device)
    fim1 = torch.fft.ifft2(torch.fft.fft2(ir_img_tensor) * Sd).real
    fim2 = torch.fft.ifft2(torch.fft.fft2(vi_img_tensor) * Sd).real
    ffim = torch.fft.ifft2(torch.fft.fft2(f_img_tensor) * Sd).real
    G1 = gaussian2d(hang, lie, 2, device).to(device)
    G2 = gaussian2d(hang, lie, 4, device).to(device)
    C1 = contrast(G1, G2, fim1)
    C2 = contrast(G1, G2, fim2)
    Cf = contrast(G1, G2, ffim)
    C1P = (k * (torch.abs(C1)**p)) / (h * (torch.abs(C1)**q) + Z)
    C2P = (k * (torch.abs(C2)**p)) / (h * (torch.abs(C2)**q) + Z)
    CfP = (k * (torch.abs(Cf)**p)) / (h * (torch.abs(Cf)**q) + Z)

    mask1 = (C1P < CfP).double()
    Q1F = (C1P / CfP) * mask1 + (CfP / C1P) * (1 - mask1)
    mask2 = (C2P < CfP).double()
    Q2F = (C2P / CfP) * mask2 + (CfP / C2P) * (1 - mask2)
    ramda1 = (C1P**2) / (C1P**2 + C2P**2 + 1e-10)
    ramda2 = (C2P**2) / (C1P**2 + C2P**2 + 1e-10)
    Q = ramda1 * Q1F + ramda2 * Q2F
    Qcb = Q.mean().item()

    return Qcb