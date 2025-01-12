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

def create_occlusion_mask(frame, direction):
    """Create an occlusion mask for the frame based on black pixels and consecutive pixel sequences.
    Args:
        frame: The frame to analyze.
        direction: The direction to process ('left' or 'right').
    Returns:
        A mask indicating the occluded regions.
    """
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        black_streak_length = 0
        if direction == 'left':
            for x in range(width):
                if np.all(frame[y, x] == [0, 0, 0]):
                    black_streak_length += 1
                    mask[y, x] = 255
                elif black_streak_length > 0:
                    non_black_streak_length = 0
                    for x2 in range(x, width):
                        if np.all(frame[y, x2] != [0, 0, 0]):
                            non_black_streak_length += 1
                        else:
                            break
                    if non_black_streak_length <= black_streak_length:
                        mask[y, x:x + non_black_streak_length] = 255
                    black_streak_length = 0
        elif direction == 'right':
            for x in range(width - 1, -1, -1):
                if np.all(frame[y, x] == [0, 0, 0]):
                    black_streak_length += 1
                    mask[y, x] = 255
                elif black_streak_length > 0:
                    non_black_streak_length = 0
                    for x2 in range(x, -1, -1):
                        if np.all(frame[y, x2] != [0, 0, 0]):
                            non_black_streak_length += 1
                        else:
                            break
                    if non_black_streak_length <= black_streak_length:
                        mask[y, x - non_black_streak_length:x] = 255
                    black_streak_length = 0

    return mask

def inpaint_test(frame, mask):
    """Paint occluded regions in the mask with blue for testing.
    Args:
        frame: The frame to modify.
        mask: The occlusion mask.
    Returns:
        Frame with occluded regions painted blue.
    """
    frame[mask == 255] = [255, 0, 0]  # Paint occluded areas blue
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

        # Create occlusion masks
        left_mask = create_occlusion_mask(left_frame, direction='left')
        right_mask = create_occlusion_mask(right_frame, direction='right')

        # Paint occluded regions for testing
        left_frame = inpaint_test(left_frame, left_mask)
        right_frame = inpaint_test(right_frame, right_mask)

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
