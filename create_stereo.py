import os
import cv2
import numpy as np
from tqdm import tqdm
import time

def luminance_from_depth(depth_map):
    """
    Calcola la luminanza normalizzata (0-1) dalla depth map.
    Depth map può essere a 8 o 16 bit.
    """
    if depth_map.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = 255
    return depth_map / max_val

def horizontal_inpaint(image, mask):
    """
    Effettua l'inpainting orizzontale su un'immagine usando una maschera.
    Riempie solo i pixel vuoti in orizzontale.
    """
    for y in range(image.shape[0]):
        row = image[y]
        row_mask = mask[y]
        indices = np.where(row_mask > 0)[0]
        if len(indices) > 0:
            for i in range(len(indices) - 1):
                start, end = indices[i], indices[i + 1]
                if end - start > 1:
                    row[start + 1:end] = np.mean([row[start], row[end]], axis=0)
    return image

def create_stereoscopic_frames(rgb_dir, depth_dir, output_dir, displacement):
    """
    Crea frame side-by-side per video stereoscopico.
    
    Parameters:
        rgb_dir (str): Directory con i frame RGB.
        depth_dir (str): Directory con le depth map.
        output_dir (str): Directory di output per i frame side-by-side.
        displacement (int): Numero di livelli/parallax offset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    total_files = len(rgb_files)
    start_time = time.time()

    for idx, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=total_files)):
        # Carica il frame RGB e la depth map
        rgb = cv2.imread(os.path.join(rgb_dir, rgb_file))
        depth = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        # Calcola la luminanza normalizzata
        luminance = luminance_from_depth(depth)

        # Crea maschere e offset per ogni livello di parallax
        height, width, _ = rgb.shape
        left_image = rgb.copy()
        right_image = rgb.copy()

        for level in range(displacement):
            lower_bound = level / displacement
            upper_bound = (level + 1) / displacement

            mask = np.logical_and(luminance >= lower_bound, luminance < upper_bound).astype(np.uint8) * 255

            # Applica l'offset per i due occhi
            left_offset = level
            right_offset = -level

            left_layer = np.zeros_like(left_image)
            right_layer = np.zeros_like(right_image)

            for y in range(height):
                for x in range(width):
                    if mask[y, x]:
                        if 0 <= x + left_offset < width:
                            left_layer[y, x + left_offset] = rgb[y, x]
                        if 0 <= x + right_offset < width:
                            right_layer[y, x + right_offset] = rgb[y, x]

            # Aggiorna le immagini con il layer corrente
            left_mask = (left_layer.sum(axis=2) == 0).astype(np.uint8)
            right_mask = (right_layer.sum(axis=2) == 0).astype(np.uint8)

            left_image = horizontal_inpaint(left_image + left_layer, left_mask)
            right_image = horizontal_inpaint(right_image + right_layer, right_mask)

        # Crea immagine side-by-side
        sbs_image = np.hstack((left_image, right_image))

        # Salva il risultato
        output_path = os.path.join(output_dir, rgb_file)
        cv2.imwrite(output_path, sbs_image)

        # Calcola il tempo rimanente
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / (idx + 1)
        remaining_time = avg_time_per_file * (total_files - (idx + 1))
        print(f"[{idx + 1}/{total_files}] Frame processato: {rgb_file}. Tempo rimanente stimato: {remaining_time:.2f} secondi.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crea video stereoscopico 3D dai frame e depth map.")
    parser.add_argument("--rgb_dir", required=True, help="Directory con i frame RGB originali.")
    parser.add_argument("--depth_dir", required=True, help="Directory con le depth map.")
    parser.add_argument("--output_dir", required=True, help="Directory per i frame side-by-side.")
    parser.add_argument("--displacement", type=int, required=True, help="Numero di livelli di parallax.")

    args = parser.parse_args()

    create_stereoscopic_frames(args.rgb_dir, args.depth_dir, args.output_dir, args.displacement)
