# CT Reconstruction using Forward and Backward Projection with PyTorch

## Project Overview
This project implements a CT image reconstruction system using Forward Projection (FP) and Filtered Back Projection (FBP) methods. It uses sparse matrices A and B to represent the relationship between projection data and images, and performs computations within the PyTorch framework. This system is useful for CT reconstruction, evaluating image quality (using PSNR), and generating projection data.

## Features
- **Generate AB Matrices**: Create sparse matrices A and B for forward and backward projection.
- **Forward Projection (FPLayer)**: Projects the CT image to generate projection data (sinogram).
- **Filtered Back Projection (FBPLayer)**: Reconstructs the CT image from the projection data.

## Code Workflow

### 1. Generate AB Matrices
Run the `Generate_AB_Matrices.py` script to generate and save the sparse matrices A and B. These matrices contain the geometric information required for forward and backward projection. After running this script once, you can comment out the matrix generation code to avoid regenerating the matrices.

```bash
python Generate_AB_Matrices.py
```

### 2. Image Reconstruction Process
Use the `CT_Reconstruction.py` script to load the original CT image (e.g., `phantom.jpg`), and perform forward projection and backward projection reconstruction via the `FPLayer` and `FBPLayer`.

The script will compute the PSNR between the original and reconstructed images and display the images.

```bash
python CT_Reconstruction.py
```

### 3. Embed FPLayer and FBPLayer into Your PyTorch Model
You can embed the `FPLayer` and `FBPLayer` as custom layers in your PyTorch deep learning model. Here's an example of how to use these layers:

```python
import torch
from FP_FBP import FPLayer, FBPLayer

# Initialize FPLayer and FBPLayer
fp = FPLayer()
fbp = FBPLayer()

# Example input CT image (sinogram)
image = torch.randn(1, 1, 256, 256)  # Replace with actual sinogram data

# Forward projection
proj_data = fp.forward(image)

# Backward projection (Reconstruction)
reconstructed_image = fbp.forward(proj_data)

# Perform evaluation (e.g., PSNR)
# Compare the reconstructed image with the original
```

## License
This project is licensed under the MIT License.
```

You can copy this into your `README.md` file in your GitHub repository.