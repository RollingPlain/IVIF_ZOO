import numpy as np
from scipy.signal import convolve2d
import math
import torch
def sobel_fn(x):
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')

    return gv, gh


def per_extn_im_fn(x, wsize):

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def get_Nabf(I1, I2, f):
    Td=2
    wt_min=0.001
    P=1
    Lg=1.5
    Nrg=0.9999
    kg=19
    sigmag=0.5
    Nra=0.9995
    ka=22
    sigmaa=0.5

    I1 = I1.cpu().numpy() if isinstance(I1, torch.Tensor) else I1
    I2 = I2.cpu().numpy() if isinstance(I2, torch.Tensor) else I2
    f = f.cpu().numpy() if isinstance(f, torch.Tensor) else f
    xrcw = f.astype(np.float64)
    x1 = I1.astype(np.float64)
    x2 = I2.astype(np.float64)

    gvA,ghA=sobel_fn(x1)
    gA=np.sqrt(ghA**2+gvA**2)

    gvB,ghB=sobel_fn(x2)
    gB=np.sqrt(ghB**2+gvB**2)

    gvF,ghF=sobel_fn(xrcw)
    gF=np.sqrt(ghF**2+gvF**2)

    gAF=np.zeros(gA.shape)
    gBF=np.zeros(gB.shape)
    aA=np.zeros(ghA.shape)
    aB=np.zeros(ghB.shape)
    aF=np.zeros(ghF.shape)
    p,q=xrcw.shape
    maskAF1 = (gA == 0) | (gF == 0)
    maskAF2 = (gA > gF)
    gAF[~maskAF1] = np.where(maskAF2, gF / gA, gA / gF)[~maskAF1]
    maskBF1 = (gB == 0) | (gF == 0)
    maskBF2 = (gB > gF)
    gBF[~maskBF1] = np.where(maskBF2, gF / gB, gB / gF)[~maskBF1]
    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan(gvA / ghA))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan(gvB / ghB))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan(gvF / ghF))

    aAF=np.abs(np.abs(aA-aF)-np.pi/2)*2/np.pi
    aBF=np.abs(np.abs(aB-aF)-np.pi/2)*2/np.pi

    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)

    wtA = wt_min * np.ones((p, q))
    wtB = wt_min * np.ones((p, q))
    cA = np.ones((p, q))
    cB = np.ones((p, q))
    wtA = np.where(gA >= Td, cA * gA ** Lg, 0)
    wtB = np.where(gB >= Td, cB * gB ** Lg, 0)

    wt_sum = np.sum(wtA + wtB)
    QAF_wtsum = np.sum(QAF * wtA) / wt_sum
    QBF_wtsum = np.sum(QBF * wtB) / wt_sum
    QABF = QAF_wtsum + QBF_wtsum


    Qdelta = np.abs(QAF - QBF)
    QCinfo = (QAF + QBF - Qdelta) / 2
    QdeltaAF = QAF - QCinfo
    QdeltaBF = QBF - QCinfo
    QdeltaAF_wtsum = np.sum(QdeltaAF * wtA) / wt_sum
    QdeltaBF_wtsum = np.sum(QdeltaBF * wtB) / wt_sum
    QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum
    QCinfo_wtsum = np.sum(QCinfo * (wtA + wtB)) / wt_sum
    QABF11 = QdeltaABF + QCinfo_wtsum

    rr = np.zeros((p, q))
    rr = np.where(gF <= np.minimum(gA, gB), 1, 0)

    LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    na1 = np.where((gF > gA) & (gF > gB), 2 - QAF - QBF, 0)
    NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

    na = np.where((gF > gA) & (gF > gB), 1, 0)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum
    return NABF