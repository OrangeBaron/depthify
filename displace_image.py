from PIL import Image, ImageChops
import numpy as np
import os
import argparse

def create_displaced_image(color_img, depth_img, levels, eye):
    # Converti la depthmap in un array numpy e normalizzalo su 0-255
    depth_array = np.array(depth_img)
    
    # Se l'immagine è a 16 bit, normalizza il range
    if depth_array.max() > 255:
        depth_array = (depth_array / 256).astype(np.uint8)  # Normalizza su 0-255

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

def create_stereo_image(color_image, depthmap_image, levels):
    # Genera l'immagine per l'occhio sinistro
    left_image = create_displaced_image(color_image, depthmap_image, levels, "left")

    # Genera l'immagine per l'occhio destro
    right_image = create_displaced_image(color_image, depthmap_image, levels, "right")

    # Ridimensiona entrambe le immagini a metà larghezza
    half_width = left_image.width // 2
    left_resized = left_image.resize((half_width, left_image.height), Image.LANCZOS)
    right_resized = right_image.resize((half_width, right_image.height), Image.LANCZOS)

    # Creazione dell'immagine stereo side-by-side
    stereo_image = Image.new("RGBA", (left_image.width, left_image.height))
    stereo_image.paste(left_resized, (0, 0))
    stereo_image.paste(right_resized, (half_width, 0))

    return stereo_image

def process_images(color_folder, depth_folder, output_folder, levels):
    # Ottieni tutti i file dalle cartelle
    color_files = sorted(os.listdir(color_folder))
    depth_files = sorted(os.listdir(depth_folder))

    # Assicurati che le cartelle contengano lo stesso numero di immagini
    if len(color_files) != len(depth_files):
        raise ValueError("Il numero di immagini nelle cartelle delle immagini e delle depthmap non corrisponde.")

    # Elabora ogni coppia di immagini
    for idx, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        # Crea il percorso completo per ogni file
        color_img_path = os.path.join(color_folder, color_file)
        depth_img_path = os.path.join(depth_folder, depth_file)

        # Carica l'immagine a colori e la depthmap (16 bit gestita)
        color_img = Image.open(color_img_path).convert("RGBA")
        depth_img = Image.open(depth_img_path)

        # Se la depthmap è a 16 bit, usa il formato "I" e normalizza
        if depth_img.mode != "I":
            depth_img = depth_img.convert("I")  # "I" è il formato per immagini a 32 bit intero (gestisce 16 bit)

        # Crea l'immagine stereo
        stereo_image = create_stereo_image(color_img, depth_img, levels)

        # Definisci il percorso di salvataggio con il nome in formato numerico
        output_path = os.path.join(output_folder, f"{idx:06d}.png")

        # Salva l'immagine stereo
        stereo_image.save(output_path, "PNG")
        print(f"Salvato {output_path}")

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea immagini 3D side-by-side stereo a partire da cartelle di immagini e depthmap.")
    parser.add_argument("color_folder", help="Cartella contenente le immagini a colori (RGB).")
    parser.add_argument("depth_folder", help="Cartella contenente le depthmap (in scala di grigi).")
    parser.add_argument("output_folder", help="Cartella per salvare le immagini stereo di output.")
    parser.add_argument("--levels", type=int, default=10, help="Numero di livelli (default: 10).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Esegui la funzione per elaborare le immagini
    process_images(args.color_folder, args.depth_folder, args.output_folder, args.levels)
