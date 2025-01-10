import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def preprocess_rgb_frame(frame):
    """Preprocess the RGB frame to avoid confusion between black pixels and holes.
    Args:
        frame: The RGB frame to preprocess.
    Returns:
        Preprocessed RGB frame where black pixels are replaced with (1, 0, 0).
    """
    black_pixel_mask = np.all(frame == [0, 0, 0], axis=2)
    frame[black_pixel_mask] = [1, 0, 0]  # Replace pure black pixels with almost black
    return frame

def create_parallax_frame(rgb_frame, depth_map, layers, factor):
    """Create a single frame with parallax effect for one eye."""
    height, width, _ = rgb_frame.shape

    # Normalize depth map to range [0, layers-1] as integers
    depth_map_normalized = cv2.normalize(depth_map, None, 0, layers - 1, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    parallax_frame = np.zeros_like(rgb_frame, dtype=np.uint8)

    for i in range(layers):
        # Create mask for pixels in the current layer
        mask = (depth_map_normalized == i)

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

        # Overlay the layer on the parallax frame, ensuring upper layers cover lower layers
        overlay_mask = np.any(layer != 0, axis=2)
        parallax_frame[overlay_mask] = layer[overlay_mask]

    return parallax_frame

def identify_and_fill_sectors(frame, direction):
    """Identify and fill sectors of black pixels for inpainting.
    Args:
        frame: The frame to inpaint.
        direction: Direction for inpainting ('left' or 'right').
    Returns:
        Inpainted frame.
    """
    height, width, _ = frame.shape
    mask = np.all(frame == 0, axis=2)  # Find black (hole) pixels

    if direction == 'left':
        for y in range(height):
            x = 0
            while x < width:
                # Start of a sector
                if mask[y, x]:
                    start = x
                    while x < width and mask[y, x]:
                        x += 1
                    end_provisional = x

                    # Analyze subsequent non-black pixels
                    non_black_count = 0
                    while x < width and not mask[y, x]:
                        x += 1
                        non_black_count += 1

                    # Check if the non-black series is longer than the black series
                    black_count = end_provisional - start
                    if non_black_count > black_count:
                        end = end_provisional
                        # Fill the sector by replicating preceding pixels
                        if start - black_count >= 0:
                            segment_to_copy = frame[y, start - black_count:start]
                            frame[y, start:end] = segment_to_copy[:end - start]
                    else:
                        # Continue to the next black sector
                        continue
                x += 1

    elif direction == 'right':
        for y in range(height):
            x = width - 1
            while x >= 0:
                # Start of a sector
                if mask[y, x]:
                    start = x
                    while x >= 0 and mask[y, x]:
                        x -= 1
                    end_provisional = x

                    # Analyze subsequent non-black pixels
                    non_black_count = 0
                    while x >= 0 and not mask[y, x]:
                        x -= 1
                        non_black_count += 1

                    # Check if the non-black series is longer than the black series
                    black_count = start - end_provisional
                    if non_black_count > black_count:
                        end = end_provisional
                        # Fill the sector by replicating following pixels
                        if start + black_count < width:
                            segment_to_copy = frame[y, start + 1:start + 1 + black_count]
                            frame[y, end + 1:start + 1] = segment_to_copy[-(start - end):]
                    else:
                        # Continue to the next black sector
                        continue
                x -= 1

    return frame

def process_frames(rgb_dir, depth_dir, output_dir, layers, factor):
    """Process all frames to create stereoscopic video."""
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')])

    assert len(rgb_files) == len(depth_files), "Mismatch in number of RGB and depth frames."

    os.makedirs(output_dir, exist_ok=True)

    for idx, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing frames")):
        # Load and preprocess RGB frame
        rgb_frame = cv2.imread(os.path.join(rgb_dir, rgb_file))
        rgb_frame = preprocess_rgb_frame(rgb_frame)

        # Load depth map
        depth_map = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        # Create parallax frames for left and right eyes
        left_frame = create_parallax_frame(rgb_frame, depth_map, layers, factor)
        right_frame = create_parallax_frame(rgb_frame, depth_map, layers, -factor)

        # Inpaint missing areas
        left_frame = identify_and_fill_sectors(left_frame, direction='left')
        right_frame = identify_and_fill_sectors(right_frame, direction='right')

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
