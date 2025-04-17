import numpy as np

def generate_AB_matrices(ImSz, D, senSpacing, rotIncr,
                         A_filename='A_matrix.npz',
                         B_filename='B_matrix.npz'):
    """
    Generate A and B matrices and save them:
      - Save A and alphaS to A_filename (keys 'A' and 'alphaS')
      - Save B, cosWeight, and fltRamp to B_filename (keys 'B', 'cosWeight', 'fltRamp')

    This function does not return any values. It only saves the relevant data to files.

    Note: If A or B are sparse matrices, they need to be converted to dense arrays before saving.
    """
    # Step 1: Generate the projection operator A and other related quantities
    A_indices, A_data, A_shape, gammaS, betaS, rotCenIm = computeA(D, ImSz, senSpacing, rotIncr)

    # Save A and alphaS to the .npz file
    np.savez(A_filename, data=A_data, indices=A_indices, shape=A_shape, gammaS=gammaS, betaS=betaS)
    print(f"Saved A, gammaS, betaS to '{A_filename}'")

    # Step 2: Compute the reconstruction matrix B, cosWeight, and fltRamp
    ProjSz = (gammaS.shape[0], betaS.shape[0])
    B_indices, B_data, B_shape, cosWeight, fltRamp = computeB(D, senSpacing, rotIncr, ProjSz, ImSz)

    # Save B properties along with cosWeight and fltRamp to the .npz file
    np.savez(B_filename, data=B_data, indices=B_indices, shape=B_shape, cosWeight=cosWeight, fltRamp=fltRamp)
    print(f"Saved B, cosWeight, fltRamp to '{B_filename}'")


def computeA(D, ImSz, senSpacing=None, rotIncr=1):
    """
    Compute sparse system matrix A and related parameters for CT reconstruction.

    Args:
        D (float): Distance (in pixel units) from source to rotation center .
        ImSz (list or tuple): Image size as [rows, cols].
        senSpacing (float, optional): Detector angle spacing in degrees. If None, defaults to (1/D)*180/pi.
        rotIncr (float, optional): Angular step size in degrees. Default is 1.

    Returns:
        A (scipy.sparse.coo_matrix): Sparse system matrix of shape (M*N, rows*cols).
        gammaS (np.ndarray): Detector angle array in radians, shape (M,).
        betaS (np.ndarray): View angle array in radians, shape (N,).
        rotCenIm (np.ndarray): Rotation center in pixel coordinates [row, col] (Start from 0).
    """
    if senSpacing is None:
        senSpacing = (1 / D) * 180 / np.pi

    # Compute rotation center in pixel coordinates (start from 1)
    ImSz = np.array(ImSz, dtype=float)
    rotCenIm = np.floor((ImSz + 1) / 2).astype(int)

    # Compute radius of image bounding circle
    circumR = np.linalg.norm(ImSz - rotCenIm + 1)

    # Calculate detector angles (alphaS) and number of detectors (M)
    deltaGamma = senSpacing * np.pi / 180.0
    numV = int(np.ceil(np.arcsin(circumR / D) / deltaGamma))
    gammaS = np.arange(-numV, numV + 1) * deltaGamma
    M = 2 * numV + 1

    # Calculate view angles (betaS) and number of views (N)
    N = int(360 / rotIncr)
    betaS = np.arange(N) * rotIncr * np.pi / 180.0

    # Generate image pixel grid
    rows, cols = int(ImSz[0]), int(ImSz[1])
    xGrid, yGrid = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))

    # Initialize lists to construct sparse matrix A
    rowList = []
    colList = []
    valList = []

    # Loop through all views and detectors to calculate ray-pixel interactions
    for view in range(N):
        beta = betaS[view]
        # Source location in image coordinates
        srcX = D * np.sin(beta) + rotCenIm[1]
        srcY = D * np.cos(beta) + rotCenIm[0]

        for detector in range(M):
            gamma = gammaS[detector]
            theta = beta - gamma  # angle between ray and y-axis

            # Compute distance from source to each pixel along the ray
            srcXGrid = srcX - xGrid
            srcYGrid = srcY - yGrid
            dGrid = np.abs(srcXGrid * np.cos(theta) - srcYGrid * np.sin(theta))

            # Compute threshold for ray passing through pixels
            th = np.abs(np.mod(theta, np.pi / 2) - np.pi / 4)
            d0 = np.cos(np.pi / 4 - th)

            # Mask for pixels intersected by ray
            msk1 = dGrid < d0
            xList = xGrid[msk1]
            yList = yGrid[msk1]
            wList = (d0 - dGrid[msk1]) / (d0 ** 2)

            # Compute row indices in projection data
            currentRow = np.full(xList.shape, detector * N + view, dtype=int)
            rowList.extend(currentRow)

            # Compute column indices in image (flattened)
            currentCol = ((yList - 1) * cols + (xList - 1)).astype(int)
            colList.extend(currentCol)

            # Add weights to value list
            valList.extend(wList.astype(np.float32))

    return np.vstack((rowList, colList)), valList, [M * N, rows * cols], \
        gammaS.astype(np.float32), betaS.astype(np.float32), rotCenIm - 1


def computeB(D, senSpacing, rotIncr, ProjSz, ImSz):
    """
    Compute sparse system matrix B and related parameters for CT reconstruction.

    Args:
        D (float): Distance from source to rotation center.
        senSpacing (float): Sensor spacing in degrees.
        rotIncr (float): Angular step size in degrees.
        ProjSz (tuple): Size of the projection data [M, N], where M is the number of projection angles and N is the
                        number of sensor elements.
        ImSz (tuple): Image size as [rows, cols].

    Returns:
        B (scipy.sparse.coo_matrix): Sparse system matrix of shape (Q, M*N), where Q is the total number of pixels in
                                    the image.
        cosWeight (np.ndarray): Weighting function for the projections, shape (M, N).
        fltRamp (np.ndarray): Ramp filter for backprojection, shape (M,).
    """

    # Get projection data dimensions M, N
    M, N = ProjSz

    # Convert angles to radians and calculate AlphaS, BetaS
    deltaGamma = senSpacing * np.pi / 180
    gammaS = (np.arange(1, M + 1) - (M + 1) / 2) * deltaGamma
    deltaBeta = rotIncr * np.pi / 180
    betaS = np.arange(N) * deltaBeta

    # Calculate cosWeight
    cosWeight = (D * deltaGamma * deltaBeta / 2) * np.cos(gammaS)[:, np.newaxis]

    # Calculate ramp filter
    fltRamp = ramp_filter(M, senSpacing)

    # Image center (rotation center)
    rotCenIm = np.floor((np.array(ImSz) + 1) / 2 - 1).astype(int)

    # Create centered grid
    x_vals = np.arange(ImSz[1]) - rotCenIm[1]
    y_vals = np.arange(ImSz[0]) - rotCenIm[0]
    xGrid, yGrid = np.meshgrid(x_vals, y_vals, indexing='xy')

    # Calculate ray source coordinates in the image center system
    srcXY = np.vstack((D * np.sin(betaS), D * np.cos(betaS)))

    # Calculate relative grid coordinates for all views
    xGridCenSrc = xGrid[..., np.newaxis] - srcXY[0, :]
    yGridCenSrc = yGrid[..., np.newaxis] - srcXY[1, :]
    d2GridAllViews = xGridCenSrc ** 2 + yGridCenSrc ** 2

    # Convert x coordinates to ray source system and compute angles
    xGridSrc = xGridCenSrc * np.cos(betaS).reshape(1, 1, -1) - yGridCenSrc * np.sin(betaS).reshape(1, 1, -1)
    gammaRadGrids = np.arcsin(xGridSrc / np.sqrt(d2GridAllViews))

    # Calculate sensor indices and interpolation weights
    gammaGridAllViews = gammaRadGrids / deltaGamma + (M + 1) / 2
    gammaGridFlr = np.floor(gammaGridAllViews).astype(int)
    val2 = (gammaGridAllViews - gammaGridFlr).ravel()
    d2 = d2GridAllViews.ravel()

    # Total number of pixels
    Q = ImSz[0] * ImSz[1]

    # Calculate row and column indices
    rowId1 = np.repeat(np.arange(Q), N)
    offsets = np.arange(N)
    colId1_all = (gammaGridFlr - 1) * N + offsets[np.newaxis, np.newaxis, :]
    colId1 = colId1_all.ravel()
    colId2 = colId1 + N

    # Calculate interpolation coefficients
    val1d2 = (1 - val2) / d2
    val2d2 = val2 / d2

    # Assemble final sparse matrix B
    rows = np.concatenate((rowId1, rowId1))
    cols = np.concatenate((colId1, colId2))
    vals = np.concatenate((val1d2, val2d2))

    return np.vstack((rows, cols)), vals.astype(np.float32), [Q, M * N], cosWeight.astype(np.float32), \
        fltRamp.astype(np.float32)


def ramp_filter(M, senSpacing):
    """
    Generate a ramp filter for backprojection in CT reconstruction.

    :param M: The size of the filter.
    :param senSpacing: Sensor spacing in degrees.
    :return: The generated ramp filter.
    """
    ns = np.arange(1, M)
    flt0 = np.zeros(2 * M - 1, dtype=float)
    center = M - 1
    flt0[center] = 1 / 4.0
    flt0[center + 1:] = (((np.cos(np.pi * ns) - 1) / ((np.pi * ns) ** 2) +
                          np.sin(np.pi * ns) / (np.pi * ns)) / 2)
    flt0[:center] = flt0[center + 1:][::-1]

    if senSpacing is not None:
        deltaAlpha = senSpacing * np.pi / 180.0
        weights = flt0.copy()
        weights[center] = 1 / (deltaAlpha ** 2)
        weights[center + 1:] = (ns / np.sin(deltaAlpha * ns)) ** 2
        weights[:center] = weights[center + 1:][::-1]
        flt = weights * flt0
    else:
        flt = flt0
    return flt
