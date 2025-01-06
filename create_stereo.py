import os
import cv2
import numpy as np
import argparse
import time

def create_stereoscopic_frames(rgb_folder, depth_folder, output_folder, displacement, factor=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(rgb_files) != len(depth_files):
        raise ValueError("The number of RGB frames and depth maps must be the same.")

    total_files = len(rgb_files)
    start_time = time.time()

    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Processing frame {i + 1}/{total_files}: {rgb_file}")

        # Load RGB and depth map
        rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
        depth_image = cv2.imread(os.path.join(depth_folder, depth_file), cv2.IMREAD_UNCHANGED)

        if depth_image is None or rgb_image is None:
            raise ValueError(f"Error loading {rgb_file} or {depth_file}.")

        # Normalize depth map to luminance values (0-255 if 8-bit, or scaled if 16-bit)
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Prepare left and right images
        height, width, _ = rgb_image.shape
        left_image = np.zeros_like(rgb_image)
        right_image = np.zeros_like(rgb_image)

        center_shift = (displacement * factor) // 2

        for level in range(displacement):
            threshold = int((level / displacement) * 255)

            # Create mask for pixels above the threshold
            mask = depth_image >= threshold

            # Shift amount for parallax effect
            shift = level * factor

            # Create left image (shift to the right with center adjustment)
            left_shifted = np.roll(rgb_image, shift - center_shift, axis=1)
            left_image[mask] = left_shifted[mask]

            # Create right image (shift to the left with center adjustment)
            right_shifted = np.roll(rgb_image, -shift + center_shift, axis=1)
            right_image[mask] = right_shifted[mask]

        # Combine left and right images side-by-side
        stereoscopic_frame = np.hstack((left_image, right_image))

        # Save the output frame
        output_path = os.path.join(output_folder, f"frame_{i + 1:06d}.png")
        cv2.imwrite(output_path, stereoscopic_frame)
        print(f"Saved: {output_path}")

        # Estimate and print remaining time
        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / (i + 1)
        remaining_time = avg_time_per_frame * (total_files - (i + 1))
        if remaining_time >= 3600:
            time_remaining_str = f"{remaining_time / 3600:.2f} hours"
        elif remaining_time >= 60:
            time_remaining_str = f"{remaining_time / 60:.2f} minutes"
        else:
            time_remaining_str = f"{remaining_time:.2f} seconds"
        print(f"Estimated time remaining: {time_remaining_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D stereoscopic frames from RGB frames and depth maps.")
    parser.add_argument("rgb_folder", help="Path to the folder containing RGB frames.")
    parser.add_argument("depth_folder", help="Path to the folder containing depth maps.")
    parser.add_argument("output_folder", help="Path to the output folder for side-by-side frames.")
    parser.add_argument("displacement", type=int, help="Number of displacement levels.")
    parser.add_argument("--factor", type=int, default=1, help="Parallax shift factor (default: 1).")

    args = parser.parse_args()

    create_stereoscopic_frames(
        rgb_folder=args.rgb_folder,
        depth_folder=args.depth_folder,
        output_folder=args.output_folder,
        displacement=args.displacement,
        factor=args.factor
    )
