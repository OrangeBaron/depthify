from PIL import Image, ImageChops
import numpy as np
import argparse

def create_displaced_image(color_image_path, depthmap_path, levels, eye):
    # Carica l'immagine a colori
    color_img = Image.open(color_image_path).convert("RGBA")
    
    # Carica la depthmap e determina il numero di bit
    depth_img = Image.open(depthmap_path)
    depth_array = np.array(depth_img)

    # Controlla se la depthmap è a 8 bit o 16 bit
    depth_max_value = 255 if depth_img.mode == "L" and depth_array.max() <= 255 else 65535
    
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

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea un'immagine 3D half-width side-by-side stereo.")
    parser.add_argument("color_image", help="Percorso dell'immagine a colori (RGB).")
    parser.add_argument("depthmap", help="Percorso della depthmap (in scala di grigi).")
    parser.add_argument("output_image", help="Percorso per salvare l'immagine stereo di output.")
    parser.add_argument("--levels", type=int, default=10, help="Numero di livelli (default: 10).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Esegui la funzione per creare l'immagine stereo
    create_stereo_image(args.color_image, args.depthmap, args.output_image, args.levels)
