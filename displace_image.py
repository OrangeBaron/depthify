from PIL import Image, ImageChops
import numpy as np
import argparse

def create_displaced_image(color_image_path, depthmap_path, output_path, levels, direction):
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
    base_shift = 1 if direction == "right" else -1

    # Applica lo shift correttivo al livello completo iniziale
    corrected_layer = ImageChops.offset(color_img, -shift_correction * base_shift, 0)
    output_img = Image.alpha_composite(output_img, corrected_layer)

    # Applica ogni livello successivo con maschera
    for i, threshold in enumerate(thresholds):
        # Crea una maschera basata sulla depthmap
        mask = (depth_array > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L")

        # Applica la maschera all'immagine a colori
        layer = Image.composite(color_img, Image.new("RGBA", color_img.size, (0, 0, 0, 0)), mask_img)

        # Calcola lo shift per il livello corrente
        shift_offset = (i + 1 - shift_correction) * base_shift
        layer = ImageChops.offset(layer, shift_offset, 0)

        # Sovrapponi il layer sull'immagine di output
        output_img = Image.alpha_composite(output_img, layer)

    # Salva l'immagine risultante
    output_img.save(output_path, "PNG")

if __name__ == "__main__":
    # Parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Crea un'immagine composita basata su depthmap.")
    parser.add_argument("color_image", help="Percorso dell'immagine a colori (RGB).")
    parser.add_argument("depthmap", help="Percorso della depthmap (in scala di grigi).")
    parser.add_argument("output_image", help="Percorso per salvare l'immagine di output.")
    parser.add_argument("--levels", type=int, default=10, help="Numero di livelli (default: 10).")
    parser.add_argument("--direction", choices=["left", "right"], default="right", help="Direzione dello shift (default: right).")

    # Leggi gli argomenti
    args = parser.parse_args()

    # Esegui la funzione
    create_displaced_image(args.color_image, args.depthmap, args.output_image, args.levels, args.direction)
