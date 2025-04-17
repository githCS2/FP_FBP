import numpy as np
from PIL import Image
from FP_FBP import FPLayer, FBPLayer
import torch
import matplotlib.pyplot as plt
from Generate_AB_Matrices import generate_AB_matrices
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库

def norm_image(I, maxV=None, minV=None):
    """
    Normalize the input image to the range [0, 255].

    :param I: The input image.
    :param maxV: The maximum value in the image. If None, it will be computed.
    :param minV: The minimum value in the image. If None, it will be computed.
    :return: The normalized image.
    """
    if maxV is None:
        maxV = np.max(I)
    if minV is None:
        minV = np.min(I)

    if maxV == minV:
        normIm = np.zeros_like(I, dtype=np.uint8)
    else:
        normIm = ((I - minV) * 255 / (maxV - minV)).astype(np.uint8)
    return normIm


def psnr(f1, f2, k=8):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images (MATLAB version ported).
    :param f1: Original image, supports any range but must be consistent with f2.
    :param f2: Reconstructed image.
    :param k: Bit depth (default is 8-bit corresponding to the range 0-255).
    :return: PSNR value.
    """
    # Compute the maximum pixel value
    fmax = 2 ** k - 1

    # Convert to float for MSE calculation
    mse = np.mean((np.array(f1, dtype=np.float64) - np.array(f2, dtype=np.float64)) ** 2)

    # Handle the case of zero MSE
    if mse == 0:
        return float('inf')

    # Compute PSNR (MATLAB formula)
    return 10 * np.log10((fmax ** 2) / mse)


# 1. Load the phantom image
I = np.array(Image.open('phantom.jpg'), dtype="float32") / 255.0

# 2. Geometric parameters
ImSz = I.shape
pixSpacing = 200 / (ImSz[0] * 11 / 16)
D = 682 / pixSpacing  # assume that the real distance from ray source to image center is 682 mm
fanDegree = np.degrees(np.arcsin(np.linalg.norm(np.array(ImSz, dtype=float) / 2 + [1.5, 1.5]) / D))
senSpacing = fanDegree / 128  # for (257, 256) sinogram
rotIncr = 360 / 256
# senSpacing = fanDegree / 187   # for (357, 360) sinogram
# rotIncr = 360 / 360

# 3. Generate A, B matrices and save them as 'A_matrix.npz' and 'B_matrix.npz'
generate_AB_matrices(ImSz, D, senSpacing, rotIncr)  # You can comment it out after running it once

# 4. Forward projection: Use the FPLayer class to load the A matrix and project the image I
# Adata = np.load('A_matrix.npz', allow_pickle=True)
# A = torch.sparse_coo_tensor(Adata['indices'],
#                             Adata['data'],
#                             tuple(Adata['shape']))
# gammaN = len(Adata['gammaS'])
# fp = FPLayer(A, gammaN)
fp = FPLayer()
projData = fp.forward(torch.tensor(I).reshape(1, 1, I.shape[0], I.shape[1]))
print(f"singram size: {projData.shape}")
# 5. FBP Reconstruction
# Bdata = np.load('B_matrix.npz', allow_pickle=True)
# B = torch.sparse_coo_tensor(Bdata['indices'],
#                             Bdata['data'],
#                             tuple(Bdata['shape']))
# fbp = FBPLayer(B, Bdata['cosWeight'], Bdata['fltRamp'])
fbp = FBPLayer()
im_rec = fbp.forward(projData)
im_rec = np.clip(im_rec, 0, 1)

# 6. Calculate PSNR and Show images
x_rec = np.array(im_rec[0][0])
print(f"PSNR: {psnr(norm_image(I), norm_image(x_rec)):.2f}")
sinoGram = np.array(projData[0][0])
if ImSz[0] > sinoGram.shape[0]:
    sinoGram_pad = np.pad(sinoGram, ((0, ImSz[0] - sinoGram.shape[0]), (0, 0)))
    I_pad = I
    x_rec_pad = x_rec
else:
    sinoGram_pad = sinoGram
    if sinoGram.shape[0] > ImSz[0]:
        I_pad = np.pad(I, ((0, sinoGram.shape[0] - ImSz[0]), (0, 0)))
        x_rec_pad = np.pad(x_rec, ((0, sinoGram.shape[0] - ImSz[0]), (0, 0)))
    else:
        I_pad = I
        x_rec_pad = x_rec
plt.figure(1)
plt.imshow(np.hstack([norm_image(I_pad), norm_image(sinoGram_pad), norm_image(x_rec_pad)]), cmap='gray')
plt.title('Reconstructed Image using FBP')
plt.axis('off')
plt.show()
