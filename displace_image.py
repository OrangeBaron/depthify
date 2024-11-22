import os
from PIL import Image, ImageChops
import numpy as np
import argparse
import glob

def create_displaced_image(color_image_path, depthmap_path, levels, eye):
    # Carica l'immagine a colori e la depthmap
    color_img = Image.open(color_image_path).convert("RGBA")
    depth_img = Image.open(depthmap_path).convert("L")

    # Convertilo in array numpy per l'elaborazione
    depth_array = np.array(depth_img)

    # Calcola gli step di soglia basati su levels
    thresholds = [(i * 255 // levels) for i in range(1, levels)]

    # Preparazione dell'immagine di output
    output_img = Image.new("RGBA", color_img.size, (0, 0, 0, 0))

    # Calcola lo shift globale correttivo
    shift_correction = (levels - 1) // 2
    base_shift = 1 if eye == "left" else -1  # Shift opposto per l'occhio sinistro/destro

    # Applica lo shift correttivo al livello completo iniziale
    corrected_layer = ImageChops.offset(color_img, shift_correction * base_shift, 0)
    output_img = Image.alpha_composite(output_img, corrected_layer)

    # Applica ogni livello successivo con maschera
    for i, threshold in enumerate(thresholds):
        # Crea una maschera basata sulla depthmap
        mask = (depth_array > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L")

        # Applica la maschera all'immagine a colori
        layer = Image.composite(color_img, Image.new("RGBA", color_img.size, (0, 0, 0, 0)), mask_img)

        # Calcola lo shift per il livello corrente
        shift_offset = (i - shift_correction) * base_shift  # Correzione applicata per allineare i livelli
        layer = ImageChops.offset(layer, shift_offset, 0)

        # Sovrapponi il layer sull'immagine di output
        output_img = Image.alpha_composite(output_img, layer)

    return output_img

def create_stereo_image(color_image_path, depthmap_path, output_path, levels):
    # Genera l'immagine per l'occhio sinistro
    left_image = create_displaced_image(color_image_path, depthmap_path, levels, "left")

    # Genera l'immagine per l'occhio destro
    right_image = create_displaced_image(color_image_path, depthmap_path, levels, "right")

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

def process_images(input_color_dir, input_depthmap_dir, output_dir, levels):
    # Assicurati che la directory di output esista
    os.makedirs(output_dir, exist_ok=True)

    # Trova tutte le immagini a colori e depthmap nelle directory
    color_images = sorted(glob.glob(os.path.join(input_color_dir, "%06d.jpg")))
    depthmaps = sorted(glob.glob(os.path.join(input_depthmap_dir, "%06d-dpt_beit_large_512.png")))

    # Elenco delle immagini a colori e depthmap corrispondenti
    for color_image, depthmap in zip(color_images, depthmaps):
        # Crea il percorso di output con la stessa numerazione
        output_image = os.path.join(output_dir, os.path.basename(color_image).replace(".jpg", ".png"))

        # Crea e salva l'immagine stereo
        create_stereo_image(color_image, depthmap, output_image, levels)

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea un'immagine 3D half-width side-by-side stereo.")
    parser.add_argument("color_image_dir", help="Directory contenente le immagini a colori (formato %06d.jpg).")
    parser.add_argument("depthmap_dir", help="Directory contenente le depthmap (formato %06d-dpt_beit_large_512.png).")
    parser.add_argument("output_dir", help="Directory di output per le immagini stereo.")
    parser.add_argument("--levels", type=int, default=10, help="Numero di livelli (default: 10).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Elabora tutte le immagini nella directory
    process_images(args.color_image_dir, args.depthmap_dir, args.output_dir, args.levels)
