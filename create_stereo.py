import os
import cv2
import numpy as np
import time

def format_time(seconds):
    if seconds > 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    elif seconds > 60:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        return f"{seconds}s"

def create_3d_stereo_frames(rgb_dir, depth_dir, output_dir, layers=10, factor=3):
    # Creazione della cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Elenco dei frame RGB e depthmap ordinati
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg'))])

    # Verifica che le cartelle abbiano lo stesso numero di file
    if len(rgb_files) != len(depth_files):
        raise ValueError("Le cartelle RGB e depthmap devono contenere lo stesso numero di file.")

    total_frames = len(rgb_files)
    start_time = time.time()

    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        frame_start_time = time.time()

        # Lettura dei file
        rgb_image = cv2.imread(os.path.join(rgb_dir, rgb_file))
        depth_image = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        # Conversione della depthmap a scala di grigi 8 bit se necessario
        if depth_image.dtype == np.uint16:
            depth_image = cv2.convertScaleAbs(depth_image, alpha=255.0 / 65535.0)

        h, w, _ = rgb_image.shape
        left_image = np.zeros_like(rgb_image)
        right_image = np.zeros_like(rgb_image)

        # Intervallo di luminanza per ogni layer
        luminance_step = 256 // layers

        # Punto di partenza per centrare i fotogrammi
        base_offset = layers * factor // 2

        for layer in range(layers):
            min_lum = layer * luminance_step
            max_lum = (layer + 1) * luminance_step

            # Maschera per selezionare i pixel del livello corrente
            mask = cv2.inRange(depth_image, min_lum, max_lum)

            # Shift per il livello corrente
            offset = base_offset - layer * factor

            # Shift per l'immagine sinistra
            left_shift = np.roll(rgb_image, -offset, axis=1)
            left_shift[:, -offset:] = 0  # Riempimento dei bordi

            # Shift per l'immagine destra
            right_shift = np.roll(rgb_image, offset, axis=1)
            right_shift[:, :offset] = 0  # Riempimento dei bordi

            # Applicazione della maschera ai livelli corrente
            left_image[mask > 0] = left_shift[mask > 0]
            right_image[mask > 0] = right_shift[mask > 0]

        # Combinazione delle immagini sinistra e destra side-by-side
        stereo_frame = np.hstack((left_image, right_image))

        # Salvataggio del frame risultante
        output_path = os.path.join(output_dir, f"{i:06d}.png")
        cv2.imwrite(output_path, stereo_frame)

        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / (i + 1)
        remaining_time = avg_time_per_frame * (total_frames - (i + 1))

        print(f"Frame {i + 1}/{total_frames} completato: {output_path} | Tempo rimanente: {format_time(int(remaining_time))}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crea frame 3D stereoscopici da immagini RGB e depthmap.")
    parser.add_argument("rgb_dir", type=str, help="Cartella contenente i frame RGB.")
    parser.add_argument("depth_dir", type=str, help="Cartella contenente le depthmap.")
    parser.add_argument("output_dir", type=str, help="Cartella di output per i frame stereoscopici.")
    parser.add_argument("--layers", type=int, default=10, help="Numero di livelli da creare.")
    parser.add_argument("--factor", type=int, default=3, help="Numero di pixel di spostamento per livello.")

    args = parser.parse_args()

    create_3d_stereo_frames(args.rgb_dir, args.depth_dir, args.output_dir, args.layers, args.factor)
