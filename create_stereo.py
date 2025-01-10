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

    # Normalize depth map to range [0, layers] as integers
    depth_map_normalized = cv2.normalize(depth_map, None, 0, layers, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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

def inpaint_horizontal_sectors(frame, direction):
    """Inpaint missing areas in the frame horizontally using sectors.
    Args:
        frame: The frame with missing areas (holes).
        direction: Direction of inpainting ('left' or 'right').
    Returns:
        Inpainted frame.
    """
    height, width, _ = frame.shape
    for y in range(height):
        x = 0 if direction == 'left' else width - 1
        step = 1 if direction == 'left' else -1

        while 0 <= x < width:
            # Identify the start of a sector (first black pixel)
            if np.all(frame[y, x] == [0, 0, 0]):
                sector_start = x
                black_count = 0
                non_black_count = 0

                # Count consecutive black and non-black pixels
                while 0 <= x < width and np.all(frame[y, x] == [0, 0, 0]):
                    black_count += 1
                    x += step

                while 0 <= x < width and not np.all(frame[y, x] == [0, 0, 0]):
                    non_black_count += 1
                    x += step

                # Check if the sector ends
                if non_black_count > black_count:
                    sector_end = x
                    sector_width = sector_end - sector_start if direction == 'left' else sector_start - sector_end

                    # Copy pixels to fill the sector
                    copy_start = sector_start - black_count * step if direction == 'left' else sector_start + black_count * step
                    copy_end = sector_start if direction == 'left' else sector_start + 1

                    if 0 <= copy_start < width and 0 <= copy_end < width:
                        fill_pixels = frame[y, copy_start:copy_end:step][::step]
                        fill_pixels = np.tile(fill_pixels, (sector_width, 1))[:sector_width]

                        if direction == 'left':
                            frame[y, sector_start:sector_end] = fill_pixels
                        else:
                            frame[y, sector_end:sector_start] = fill_pixels[::-1]

            x += step

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

        # Inpaint missing areas using sectors
        left_frame = inpaint_horizontal_sectors(left_frame, direction='left')
        right_frame = inpaint_horizontal_sectors(right_frame, direction='right')

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
