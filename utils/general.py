import numpy as np
import os
import torch
import SimpleITK as sitk
import pandas as pd

import torch
import numpy as np
import SimpleITK as sitk


def load_NLST_data(patient_id, data_root="/home/home/johannes_julian/NLST_red/NLST"):
    """
    Loads images, masks, and landmarks for a specific patient from the NLST dataset.
    
    Args:
        patient_id (int): The patient ID (e.g., 1 for NLST_0001).
        data_root (str): The root directory of the NLST dataset.
        
    Returns:
        tuple: A tuple containing the inspiration image, expiration image,
               inspiration landmarks, expiration landmarks, mask, and voxel size.
    """
    
    # Construct file paths based on patient_id
    patient_dir_name = f"NLST_{patient_id:04d}"
    image_dir = os.path.join(data_root, "imagesTr")
    keypoints_dir = os.path.join(data_root, "keypointsTr")
    masks_dir = os.path.join(data_root, "masksTr")

    insp_path = os.path.join(image_dir, f"{patient_dir_name}_0000.nii.gz")
    exp_path = os.path.join(image_dir, f"{patient_dir_name}_0001.nii.gz")
    insp_mask_path = os.path.join(masks_dir, f"{patient_dir_name}_0000.nii.gz")
    
    # Load images and masks using SimpleITK
    insp_image_sitk = sitk.ReadImage(insp_path)
    exp_image_sitk = sitk.ReadImage(exp_path)
    mask_sitk = sitk.ReadImage(insp_mask_path)
    
    # Get voxel size (spacing) and image size (shape) from SimpleITK objects
    voxel_size = insp_image_sitk.GetSpacing()
    
    # Convert SimpleITK images to numpy arrays
    image_insp = sitk.GetArrayFromImage(insp_image_sitk)
    image_exp = sitk.GetArrayFromImage(exp_image_sitk)
    mask = sitk.GetArrayFromImage(mask_sitk)
    
    # Convert numpy arrays to PyTorch tensors
    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)
    mask = torch.FloatTensor(mask)
    
    # Load landmarks from CSV files
    landmarks_insp_path = os.path.join(keypoints_dir, f"{patient_dir_name}_0000.csv")
    landmarks_exp_path = os.path.join(keypoints_dir, f"{patient_dir_name}_0001.csv")

    landmarks_insp = pd.read_csv(landmarks_insp_path).to_numpy()
    landmarks_exp = pd.read_csv(landmarks_exp_path).to_numpy()
    
    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        list(voxel_size)  # Convert tuple to list for consistency
    )


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_landmarks(network, landmarks_pre, image_size):

    print(f"Image Size:{image_size}")

    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    print(coordinate_tensor.shape)

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta



def compute_deformation_field(network, image_size, voxel_size=(1.0, 1.0, 1.0), output_path=None):
    """
    Compute deformation field for all voxels in the image using a neural network.
    
    Args:
        network: The neural network model
        image_size: Tuple of image dimensions (e.g., (256, 256, 256))
        voxel_size: Tuple of voxel spacing (e.g., (1.0, 1.0, 1.0) in mm)
        output_path: Optional path to save the deformation field as NIfTI
    
    Returns:
        deformed_coords: Deformed coordinates
        delta: Displacement field
    """
    
    print(f"Image Size: {image_size}")
    
    # Create coordinate grids for each dimension
    coords = np.meshgrid(
        np.arange(image_size[0]),
        np.arange(image_size[1]), 
        np.arange(image_size[2]),
        indexing='ij'
    )
    
    # Stack coordinates into a single array
    # Shape: (H, W, D, 3)
    coordinate_grid = np.stack(coords, axis=-1)
    
    # Reshape to (N, 3) where N is the total number of voxels
    original_shape = coordinate_grid.shape
    coordinates = coordinate_grid.reshape(-1, 3)
    
    print(f"Total coordinates: {coordinates.shape[0]}")
    
    # Normalize coordinates to [-1, 1]
    scale_of_axes = np.array([(0.5 * s) for s in image_size])
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"Scale_of_axes: {scale_of_axes}")
    
    # Ensure proper broadcasting for normalization
    coordinate_tensor = torch.FloatTensor(coordinates / scale_of_axes[np.newaxis, :]) - 1.0
    
    print(f"Coordinate tensor shape: {coordinate_tensor.shape}")
    
    # Process in batches to avoid memory issues
    batch_size = 50000  # Adjust based on your GPU memory
    all_outputs = []
    
    network.eval()
    with torch.no_grad():
        for i in range(0, len(coordinate_tensor), batch_size):
            batch = coordinate_tensor[i:i+batch_size]
            batch_output = network(batch.cuda())
            all_outputs.append(batch_output.cpu())
    
    # Concatenate all outputs
    output = torch.cat(all_outputs, dim=0)
    print(f"Network output shape: {output.shape}")  # Debug print
    
    # Convert back to numpy and scale
    delta_raw = output.detach().numpy()
    print(f"Raw delta shape: {delta_raw.shape}")
    print(f"Scale_of_axes shape: {scale_of_axes.shape}")
    
    # Ensure proper broadcasting
    delta = delta_raw * scale_of_axes[np.newaxis, :]
    print(f"Delta shape after scaling: {delta.shape}")  # Debug print
    
    # Reshape delta back to original grid shape
    delta_field = delta.reshape(original_shape)
    print(f"Delta field shape after reshape: {delta_field.shape}")
    print(f"Expected delta field shape: {original_shape}")
    
    # Compute deformed coordinates
    deformed_coords = coordinates + delta
    deformed_grid = deformed_coords.reshape(original_shape)
    print(f"Deformed grid shape: {deformed_grid.shape}")
    
    # Save as NIfTI if path is provided
    if output_path:
        save_deformation_field_nifti(delta_field, output_path, image_size, voxel_size)
    
    return deformed_grid, delta_field

def save_deformation_field_nifti(delta_field, output_path, image_size, voxel_size):
    """
    Save the deformation field as a NIfTI file using SimpleITK.
    
    Args:
        delta_field: Displacement field of shape (H, W, D, 3)
        output_path: Path to save the NIfTI file
        image_size: Original image dimensions
        voxel_size: Tuple of voxel spacing (e.g., (1.0, 1.0, 1.0) in mm)
    """
    
    print(f"Input delta_field shape: {delta_field.shape}")
    print(f"Expected shape: {image_size + (3,)}")
    
    # For SimpleITK vector fields, we need to create it differently
    # SimpleITK expects vector fields as separate components
    print(f"Input delta_field shape: {delta_field.shape}")
    
    # Split into individual components
    dx = delta_field[:, :, :, 0]  # X displacement
    dy = delta_field[:, :, :, 1]  # Y displacement  
    dz = delta_field[:, :, :, 2]  # Z displacement
    
    print(f"Individual component shapes: {dx.shape}, {dy.shape}, {dz.shape}")
    
    # Create SimpleITK images for each component
    dx_img = sitk.GetImageFromArray(dx)
    dy_img = sitk.GetImageFromArray(dy) 
    dz_img = sitk.GetImageFromArray(dz)
    
    # Compose into vector image
    sitk_image = sitk.Compose([dx_img, dy_img, dz_img])
    
    print(f"SimpleITK image size: {sitk_image.GetSize()}")
    print(f"SimpleITK image components per pixel: {sitk_image.GetNumberOfComponentsPerPixel()}")
    
    # Set spacing (SimpleITK uses (x, y, z) order, which is reversed from numpy)
    sitk_image.SetSpacing(voxel_size[::-1])  # Reverse to match SimpleITK convention
    
    # Set origin to (0, 0, 0) - adjust if needed
    sitk_image.SetOrigin([0.0, 0.0, 0.0])
    
    # Set direction matrix to identity - adjust if needed
    sitk_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    
    # Save the file
    sitk.WriteImage(sitk_image, output_path)
    print(f"Deformation field saved to: {output_path}")
    print(f"Final image spacing: {sitk_image.GetSpacing()}")
    print(f"Final image size: {sitk_image.GetSize()}")
    print(f"Final vector components: {sitk_image.GetNumberOfComponentsPerPixel()}")

def load_image_DIRLab(variation=1, folder=r"D:\Data\DIRLAB\Case"):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    with open(folder + r"Images\case" + str(variation) + "_T00_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

    with open(folder + r"Images\case" + str(variation) + "_T50_s.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

    imgsitk_in = sitk.ReadImage(folder + r"Masks\case" + str(variation) + "_T00_s.mhd")

    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + r"ExtremePhases\Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + r"ExtremePhases\Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor
