import os
import glob
from PIL import Image, ImageChops
import numpy as np
import argparse

def create_displaced_image(color_image_path, depthmap_path, levels, eye):
    # Carica l'immagine a colori
    color_img = Image.open(color_image_path).convert("RGBA")
    
    # Carica la depthmap correttamente
    depth_img = Image.open(depthmap_path)
    
    # Verifica se l'immagine depth è in formato a 16 bit o 8 bit
    if depth_img.mode == "I" or depth_img.mode == "I;16":
        depth_array = np.array(depth_img, dtype=np.uint16)  # Depthmap a 16 bit
        depth_max_value = 65535  # Max per depthmap a 16 bit
    elif depth_img.mode == "L":
        depth_array = np.array(depth_img, dtype=np.uint8)  # Depthmap a 8 bit
        depth_max_value = 255  # Max per depthmap a 8 bit
    else:
        raise ValueError("Formato depthmap non supportato: deve essere 8 bit o 16 bit")

    # Calcola gli step di soglia basati su levels
    thresholds = [(i * depth_max_value // levels) for i in range(1, levels)]

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

def process_images(color_folder, depthmap_folder, output_folder, levels):
    # Trova tutti i file di immagini e depthmap nelle cartelle di input
    color_files = sorted(glob.glob(os.path.join(color_folder, "*.jpg")))
    depthmap_files = sorted(glob.glob(os.path.join(depthmap_folder, "*.png")))

    # Verifica che ci siano lo stesso numero di immagini e depthmap
    if len(color_files) != len(depthmap_files):
        raise ValueError("Il numero di immagini a colori e depthmap non corrisponde.")

    # Per ogni coppia di immagini e depthmap
    for i, (color_file, depthmap_file) in enumerate(zip(color_files, depthmap_files)):
        # Estrai il numero dalla base del nome del file (es. 000001)
        base_name = os.path.splitext(os.path.basename(color_file))[0]

        # Definisci il percorso di output per ogni immagine stereo
        output_file = os.path.join(output_folder, f"{base_name}.png")

        # Crea l'immagine stereo per questa coppia e salvala
        create_stereo_image(color_file, depthmap_file, output_file, levels)
        print(f"Elaborato {base_name}")

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea immagini stereo da immagini a colori e depthmap.")
    parser.add_argument("color_folder", help="Cartella contenente le immagini a colori (jpg).")
    parser.add_argument("depthmap_folder", help="Cartella contenente le depthmap (png).")
    parser.add_argument("output_folder", help="Cartella di output per le immagini stereo.")
    parser.add_argument("--levels", type=int, default=24, help="Numero di livelli (default: 24).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Crea la cartella di output se non esiste
    os.makedirs(args.output_folder, exist_ok=True)

    # Elabora le immagini
    process_images(args.color_folder, args.depthmap_folder, args.output_folder, args.levels)
