import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def create_parallax_frame(rgb_frame, depth_map, layers, factor):
    """Create a single frame with parallax effect for one eye."""
    height, width, _ = rgb_frame.shape

    # Normalize depth map to range [0.0, 1.0]
    if depth_map.dtype == np.uint8:
        depth_map_normalized = depth_map.astype(np.float32) / 255.0
    elif depth_map.dtype == np.uint16:
        depth_map_normalized = depth_map.astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported depth map format. Expected 8-bit or 16-bit images.")

    # Create an empty parallax frame and boolean mask for filled pixels
    parallax_frame = np.zeros_like(rgb_frame, dtype=np.uint8)
    pixel_set_mask = np.zeros((height, width), dtype=bool)

    for i in range(layers):
        min_luminance = i / layers
        max_luminance = (i + 1) / layers

        # Create mask for pixels in the current luminance range
        mask = (depth_map_normalized >= min_luminance) & (depth_map_normalized < max_luminance) & (~pixel_set_mask)

        # Extract the layer and shift it horizontally
        layer = np.zeros_like(rgb_frame, dtype=np.uint8)
        layer[mask] = rgb_frame[mask]

        shift = int((i - layers / 2) * factor)
        if shift > 0:
            layer = np.roll(layer, shift, axis=1)
            layer[:, :shift] = 0
        elif shift < 0:
            layer = np.roll(layer, shift, axis=1)
            layer[:, shift:] = 0

        # Overlay the shifted layer into the parallax frame where not already filled
        overlay_mask = mask & ~pixel_set_mask
        parallax_frame[overlay_mask] = layer[overlay_mask]
        pixel_set_mask |= mask

    return parallax_frame

def inpaint_horizontal(frame, direction):
    """Inpaint missing areas in the frame horizontally using NumPy."""
    mask = np.all(frame == 0, axis=2)  # Find holes (where all RGB values are 0)

    if direction == 'left':
        for x in range(1, frame.shape[1]):
            frame[:, x][mask[:, x]] = frame[:, x - 1][mask[:, x]]
    elif direction == 'right':
        for x in range(frame.shape[1] - 2, -1, -1):
            frame[:, x][mask[:, x]] = frame[:, x + 1][mask[:, x]]

    return frame

def process_frames(rgb_dir, depth_dir, output_dir, layers, factor, deflicker):
    """Process all frames to create stereoscopic video."""
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')])

    assert len(rgb_files) == len(depth_files), "Mismatch in number of RGB and depth frames."

    os.makedirs(output_dir, exist_ok=True)

    for idx, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing frames")):
        # Load RGB frame
        rgb_frame = cv2.imread(os.path.join(rgb_dir, rgb_file))

        # Load depth maps and apply deflicker if enabled
        depth_map = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)
        if deflicker:
            prev_depth_map = cv2.imread(os.path.join(depth_dir, depth_files[idx - 1]), cv2.IMREAD_UNCHANGED) if idx > 0 else depth_map
            next_depth_map = cv2.imread(os.path.join(depth_dir, depth_files[idx + 1]), cv2.IMREAD_UNCHANGED) if idx < len(depth_files) - 1 else depth_map
            depth_map = np.mean([prev_depth_map, depth_map, next_depth_map], axis=0).astype(depth_map.dtype)

        # Create parallax frames for left and right eyes
        left_frame = create_parallax_frame(rgb_frame, depth_map, layers, factor)
        right_frame = create_parallax_frame(rgb_frame, depth_map, layers, -factor)

        # Inpaint missing areas
        left_frame = inpaint_horizontal(left_frame, direction='left')
        right_frame = inpaint_horizontal(right_frame, direction='right')

        # Combine frames side-by-side
        side_by_side_frame = np.hstack((left_frame, right_frame))

        # Save the combined frame
        output_path = os.path.join(output_dir, f"{idx:06d}.png")
        cv2.imwrite(output_path, side_by_side_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 3D stereoscopic video frames from RGB frames and depth maps.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB frames.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth maps.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output frames.")
    parser.add_argument("--layers", type=int, default=10, help="Number of parallax layers.")
    parser.add_argument("--factor", type=int, default=3, help="Parallax shift factor.")
    parser.add_argument("--deflicker", action='store_true', help="Apply deflicker by averaging depth maps with adjacent frames.")

    args = parser.parse_args()

    process_frames(args.rgb_dir, args.depth_dir, args.output_dir, args.layers, args.factor, args.deflicker)
