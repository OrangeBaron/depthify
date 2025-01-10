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

def fill_sectors(frame, direction):
    """Fill sectors of missing pixels in the frame based on the specified direction.
    Args:
        frame: The frame with missing areas (holes).
        direction: Direction to process ('left' or 'right').
    Returns:
        Frame with sectors filled.
    """
    height, width, _ = frame.shape
    mask = np.all(frame == 0, axis=2)  # Find holes (where all RGB values are 0)

    if direction == 'left':
        for y in range(height):
            x = 0
            while x < width:
                if mask[y, x]:
                    # Start of a sector
                    start = x

                    # Measure consecutive black pixels
                    while x < width and mask[y, x]:
                        x += 1
                    end = x  # Provisional end

                    # Measure consecutive non-black pixels after the provisional end
                    non_black_start = end
                    while x < width and not mask[y, x]:
                        x += 1
                    non_black_length = x - non_black_start

                    # Determine the true end of the sector
                    if non_black_length >= (end - start):
                        true_end = end
                    else:
                        continue  # Skip the sector if the non-black region is too short

                    # Fill the sector
                    fill_start = max(0, start - (true_end - start))
                    if fill_start < start:
                        frame[y, start:true_end] = frame[y, fill_start:start]

    elif direction == 'right':
        for y in range(height):
            x = width - 1
            while x >= 0:
                if mask[y, x]:
                    # Start of a sector
                    start = x

                    # Measure consecutive black pixels
                    while x >= 0 and mask[y, x]:
                        x -= 1
                    end = x  # Provisional end

                    # Measure consecutive non-black pixels before the provisional end
                    non_black_start = end
                    while x >= 0 and not mask[y, x]:
                        x -= 1
                    non_black_length = non_black_start - x

                    # Determine the true end of the sector
                    if non_black_length >= (start - end):
                        true_end = end
                    else:
                        continue  # Skip the sector if the non-black region is too short

                    # Fill the sector
                    fill_end = min(width - 1, start + (start - true_end))
                    if fill_end > start:
                        frame[y, true_end:start] = frame[y, start:fill_end]

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

        # Fill sectors
        left_frame = fill_sectors(left_frame, direction='left')
        right_frame = fill_sectors(right_frame, direction='right')

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
