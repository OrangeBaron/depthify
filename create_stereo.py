import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def create_parallax_frame(rgb_frame, depth_map, layers, factor):
    """Create a single frame with parallax effect for one eye."""
    height, width, _ = rgb_frame.shape

    # Normalize depth map to range [0.0, 1.0]
    depth_map_normalized = cv2.normalize(depth_map.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    parallax_frame = np.zeros_like(rgb_frame, dtype=np.uint8)

    for i in range(layers):
        min_luminance = i / layers
        max_luminance = (i + 1) / layers

        # Create mask for pixels in the current luminance range
        mask = (depth_map_normalized >= min_luminance) & (depth_map_normalized < max_luminance)

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

        # Accumulate the shifted layer into the parallax frame
        parallax_frame = cv2.add(parallax_frame, layer)

    return parallax_frame

def inpaint_horizontal(frame, direction):
    """Inpaint missing areas in the frame horizontally.
    Args:
        frame: The frame with missing areas (holes).
        direction: Direction of inpainting ('left' or 'right').
    Returns:
        Inpainted frame.
    """
    mask = np.all(frame == 0, axis=2)  # Find holes (where all RGB values are 0)

    if direction == 'left':
        for y in range(frame.shape[0]):
            for x in range(1, frame.shape[1]):
                if mask[y, x]:
                    frame[y, x] = frame[y, x - 1]  # Fill from the left
    elif direction == 'right':
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1] - 2, -1, -1):
                if mask[y, x]:
                    frame[y, x] = frame[y, x + 1]  # Fill from the right

    return frame

def process_frames(rgb_dir, depth_dir, output_dir, layers, factor):
    """Process all frames to create stereoscopic video."""
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')])

    assert len(rgb_files) == len(depth_files), "Mismatch in number of RGB and depth frames."

    os.makedirs(output_dir, exist_ok=True)

    for idx, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing frames")):
        # Load RGB frame and depth map
        rgb_frame = cv2.imread(os.path.join(rgb_dir, rgb_file))
        depth_map = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

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

    args = parser.parse_args()

    process_frames(args.rgb_dir, args.depth_dir, args.output_dir, args.layers, args.factor)
