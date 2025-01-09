import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

def create_parallax_frame_gpu(rgb_frame, depth_map, layers, factor, device):
    """Create a single frame with parallax effect for one eye using GPU."""
    # Convert to PyTorch tensors and move to device
    rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().to(device)  # Shape: (C, H, W)
    depth_tensor = torch.from_numpy(depth_map).float().to(device)  # Shape: (H, W)

    # Normalize depth map to range [0.0, 1.0]
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

    parallax_frame = torch.zeros_like(rgb_tensor)  # Initialize the output frame

    for i in range(layers):
        min_luminance = i / layers
        max_luminance = (i + 1) / layers

        # Create mask for the current layer
        mask = (depth_tensor >= min_luminance) & (depth_tensor < max_luminance)

        # Expand mask to RGB channels and apply it
        mask_rgb = mask.unsqueeze(0).expand_as(rgb_tensor)  # Shape: (C, H, W)
        layer = torch.where(mask_rgb, rgb_tensor, torch.zeros_like(rgb_tensor))

        # Compute horizontal shift
        shift = int((i - layers / 2) * factor)
        if shift > 0:
            layer = torch.roll(layer, shifts=shift, dims=2)  # Shift to the right
            layer[:, :, :shift] = 0  # Clear wrapped pixels
        elif shift < 0:
            layer = torch.roll(layer, shifts=shift, dims=2)  # Shift to the left
            layer[:, :, shift:] = 0  # Clear wrapped pixels

        # Overlay the layer on the parallax frame
        overlay_mask = mask.unsqueeze(0).expand_as(rgb_tensor)
        parallax_frame = torch.where(overlay_mask, layer, parallax_frame)

    # Convert back to numpy and uint8
    parallax_frame = parallax_frame.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return parallax_frame

def process_frames_gpu(rgb_dir, depth_dir, output_dir, layers, factor, device):
    """Process all frames to create stereoscopic video using GPU."""
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')])

    assert len(rgb_files) == len(depth_files), "Mismatch in number of RGB and depth frames."

    os.makedirs(output_dir, exist_ok=True)

    for idx, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing frames")):
        # Load RGB frame and depth map
        rgb_frame = cv2.imread(os.path.join(rgb_dir, rgb_file))
        depth_map = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        # Create parallax frames for left and right eyes
        left_frame = create_parallax_frame_gpu(rgb_frame, depth_map, layers, factor, device)
        right_frame = create_parallax_frame_gpu(rgb_frame, depth_map, layers, -factor, device)

        # Combine frames side-by-side
        side_by_side_frame = np.hstack((left_frame, right_frame))

        # Save the combined frame
        output_path = os.path.join(output_dir, f"{idx:06d}.png")
        cv2.imwrite(output_path, side_by_side_frame)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create 3D stereoscopic video frames from RGB frames and depth maps using GPU.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB frames.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth maps.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output frames.")
    parser.add_argument("--layers", type=int, default=10, help="Number of parallax layers.")
    parser.add_argument("--factor", type=int, default=3, help="Parallax shift factor.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    process_frames_gpu(args.rgb_dir, args.depth_dir, args.output_dir, args.layers, args.factor, device)
