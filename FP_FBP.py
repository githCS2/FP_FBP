import numpy as np
import torch
import torch.nn as nn

# import scipy
# from scipy.sparse import coo_matrix
# from scipy.signal import convolve2d

class FPLayer(nn.Module):
    """
    Forward Projection (FP):
      - By default, loads the sparse matrix A (COO format) and projection angles alphaS from the "A_matrix.npz" file
      - Projects the input image I while preserving the Fortran order)
    """

    def __init__(self, A=None, gammaN=None):
        super(FPLayer, self).__init__()
        if A is None:
            A_filename = "A_matrix.npz"  # Default filename
            data = np.load(A_filename, allow_pickle=True)
            # Reconstruct the sparse matrix A
            self.A = torch.sparse_coo_tensor(data['indices'],
                                             data['data'],
                                             tuple(data['shape'])
                                             )
            self.gammaN = len(data['gammaS'])
        else:
            self.A = A
            self.gammaN = gammaN

    def forward(self, x):
        """
        Perform forward projection on image I

        Parameters:
          x: Input image (2D numpy array)

        Returns:
          Projection data, with shape (len(alphaS), n_cols)
        """
        # Flatten the image in Fortran order (column-major)
        B, C, N, K = x.shape
        x2D = x.reshape(B * C, K * N)

        proj2D = torch.sparse.mm(self.A, x2D.t())
        return proj2D.reshape(B, C, self.gammaN, -1)


class FBPLayer(nn.Module):
    """
    Filtered Backprojection (FBP):
      - By default, loads the sparse matrix B (COO format), cosWeight, and fltRamp from the "B_matrix.npz" file
      - Applies weighting, filtering, and then performs backprojection to reconstruct the image using matrix B
    """

    def __init__(self, B=None, cosWeight=None, fltRamp=None, imSz=None):
        super(FBPLayer, self).__init__()
        if B is None:
            B_filename = "B_matrix.npz"  # Default filename
            data = np.load(B_filename, allow_pickle=True)
            # Reconstruct the sparse matrix B
            self.B = torch.sparse_coo_tensor(data['indices'],
                                             data['data'],
                                             tuple(data['shape']))
            if cosWeight is None:
                self.cosWeight = torch.tensor(data['cosWeight'])
            else:
                self.cosWeight = torch.tensor(cosWeight)
            if fltRamp is None:
                self.fltRamp = torch.tensor(data['fltRamp'])
            else:
                self.fltRamp = torch.tensor(fltRamp)
        else:
            self.B = B
            self.fltRamp = torch.tensor(fltRamp)
            self.cosWeight = torch.tensor(cosWeight)

        if imSz is None:
            sz = int(self.B.shape[0] ** 0.5)
            self.imSz = (sz, sz)
        else:
            self.imSz = imSz

    def forward(self, projData):
        """
        Apply weighting, filtering, and backprojection on projection data projDataA, returning the reconstructed image (clipped to [0, 1])

        Parameters:
          projData: Projection data, 2D numpy array

        Returns:
          Reconstructed image
        """
        B, C, N, K = projData.shape
        # 1) Apply weighting
        proj_w = projData * self.cosWeight.reshape(-1, 1)
        # 2) Filter the data using 2D convolution
        kernel = self.fltRamp.reshape(1, 1, -1, 1)
        if kernel.shape[2] > 1:
            proj_f = nn.functional.conv2d(proj_w, kernel, padding='same')
        else:
            proj_f = proj_w

        # 3) Backprojection
        proj2D = proj_f.reshape(B * C, K * N)
        x2D = torch.sparse.mm(self.B, proj2D.t())
        return x2D.reshape(B, C, -1, self.imSz[1])