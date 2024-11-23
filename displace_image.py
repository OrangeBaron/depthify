from PIL import Image, ImageChops
import numpy as np
import argparse
import os
import time

def load_depthmap(depthmap_path):
    """
    Carica una depthmap che può essere a 8 o 16 bit.
    Ritorna un array numpy normalizzato a 8 bit (0-255).
    """
    depth_img = Image.open(depthmap_path)
    
    if depth_img.mode == "I;16":  # 16-bit depthmap
        depth_array = np.array(depth_img, dtype=np.uint16)
        depth_array = (depth_array / 65535.0 * 255).astype(np.uint8)  # Normalizza a 8 bit
    elif depth_img.mode == "L":  # 8-bit depthmap
        depth_array = np.array(depth_img, dtype=np.uint8)
    else:
        raise ValueError("Unsupported depthmap format. Use 8-bit or 16-bit grayscale PNG.")
    
    return depth_array

def calculate_thresholds(levels):
    """
    Calcola le soglie per ciascun livello in base al numero di livelli.

    Parameters:
        levels (int): Numero di livelli.

    Returns:
        list[int]: Lista di soglie (valori di luminanza) per ciascun livello.
    """
    return [(i * 255 // levels) for i in range(levels)]

def create_displaced_image(color_image_path, depthmap_path, levels, eye, factor):
    # Carica l'immagine a colori
    color_img = Image.open(color_image_path).convert("RGBA")

    # Carica e normalizza la depthmap
    depth_array = load_depthmap(depthmap_path)

    # Calcola gli step di soglia basati su levels
    thresholds = calculate_thresholds(levels)

    # Preparazione dell'immagine di output
    output_img = Image.new("RGBA", color_img.size, (0, 0, 0, 0))

    # Calcola lo shift globale correttivo
    shift_correction = (levels - 1) // 2 * factor
    base_shift = 1 if eye == "left" else -1  # Correzione dello shift base

    # Applica ogni livello con maschera
    for i, threshold in enumerate(thresholds):
        # Crea una maschera basata sulla depthmap
        mask = (depth_array >= threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L")

        # Applica la maschera all'immagine a colori
        layer = Image.composite(color_img, Image.new("RGBA", color_img.size, (0, 0, 0, 0)), mask_img)

        # Calcola lo shift per il livello corrente
        shift_offset = (i - shift_correction) * base_shift * factor
        layer = ImageChops.offset(layer, shift_offset, 0)

        # Sovrapponi il layer sull'immagine di output
        output_img = Image.alpha_composite(output_img, layer)

    return output_img

def create_stereo_image(color_image_path, depthmap_path, output_path, levels, factor):
    # Genera l'immagine per l'occhio sinistro
    left_image = create_displaced_image(color_image_path, depthmap_path, levels, "left", factor)

    # Genera l'immagine per l'occhio destro
    right_image = create_displaced_image(color_image_path, depthmap_path, levels, "right", factor)

    # Ridimensiona entrambe le immagini a metà larghezza
    half_width = left_image.width // 2
    left_resized = left_image.resize((half_width, left_image.height), Image.LANCZOS)
    right_resized = right_image.resize((half_width, right_image.height), Image.LANCZOS)

    # Creazione dell'immagine stereo side-by-side
    stereo_image = Image.new("RGBA", (left_image.width, left_image.height))
    stereo_image.paste(left_resized, (0, 0))
    stereo_image.paste(right_resized, (half_width, 0))

    # Salva l'immagine risultante
    stereo_image.save(output_path, "PNG")

def process_folder(color_dir, depth_dir, output_dir, levels, factor):
    # Ottieni la lista di file nell'ordine corretto
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    # Assicurati che la cartella di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Calcola il numero totale di immagini
    num_images = len(color_files)
    if num_images == 0:
        print("No images found in the input directories.")
        return

    # Inizializza il timer
    start_time = time.time()

    # Processa ogni file
    for index, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        # Costruisci i percorsi completi
        color_path = os.path.join(color_dir, color_file)
        depth_path = os.path.join(depth_dir, depth_file)
        output_path = os.path.join(output_dir, color_file.replace(".jpg", ".png"))

        print(f"Processing {index + 1}/{num_images}: {color_file} and {depth_file} -> {output_path}")

        # Stima del tempo rimanente
        elapsed_time = time.time() - start_time
        images_processed = index + 1
        images_remaining = num_images - images_processed
        avg_time_per_image = elapsed_time / images_processed if images_processed > 0 else 0
        estimated_time_remaining = avg_time_per_image * images_remaining

        if elapsed_time < 60:
            elapsed_time_str = "{:.0f}s".format(elapsed_time)
        elif elapsed_time < 3600:
            elapsed_time_str = "{:.0f}m".format(elapsed_time / 60)
        else:
            elapsed_time_str = "{:.0f}h".format(elapsed_time / 3600)

        if estimated_time_remaining < 60:
            time_remaining_str = "{:.0f}s".format(estimated_time_remaining)
        elif estimated_time_remaining < 3600:
            time_remaining_str = "{:.0f}m".format(estimated_time_remaining / 60)
        else:
            time_remaining_str = "{:.0f}h".format(estimated_time_remaining / 3600)

        print(f"    Time elapsed: {elapsed_time_str} | Estimated time remaining: {time_remaining_str}")

        # Crea l'immagine stereo
        create_stereo_image(color_path, depth_path, output_path, levels, factor)

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea immagini 3D stereo da cartelle di immagini e depthmap.")
    parser.add_argument("color_dir", help="Cartella contenente le immagini a colori (es. 000001.jpg).")
    parser.add_argument("depth_dir", help="Cartella contenente le depthmap (es. 000001.png).")
    parser.add_argument("output_dir", help="Cartella per salvare le immagini stereo.")
    parser.add_argument("--levels", type=int, default=10, help="Numero di livelli (default: 10).")
    parser.add_argument("--factor", type=int, default=1, help="Fattore di amplificazione dello shift (default: 1).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Processa i file nelle cartelle
    process_folder(args.color_dir, args.depth_dir, args.output_dir, args.levels, args.factor)
