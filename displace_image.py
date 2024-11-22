from PIL import Image, ImageChops
import numpy as np

def create_displaced_image(color_image_path, depthmap_path, output_path, N, direction):
    # Carica l'immagine a colori e la depthmap
    color_img = Image.open(color_image_path).convert("RGBA")
    depth_img = Image.open(depthmap_path).convert("L")

    # Convertilo in array numpy per l'elaborazione
    color_array = np.array(color_img)
    depth_array = np.array(depth_img)

    # Calcola gli step di soglia basati su N
    thresholds = [(i * 255 // N) for i in range(N)]

    # Preparazione dell'immagine di output
    output_img = Image.new("RGBA", color_img.size, (0, 0, 0, 0))

    # Definisci la direzione dello shift
    shift = (1, 0) if direction == "right" else (-1, 0)

    # Applica ogni livello
    for i, threshold in enumerate(thresholds):
        # Crea una maschera basata sulla depthmap
        mask = (depth_array > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L")

        # Applica la maschera all'immagine a colori
        layer = Image.composite(color_img, Image.new("RGBA", color_img.size, (0, 0, 0, 0)), mask_img)

        # Shift del layer
        layer = ImageChops.offset(layer, i * shift[0], i * shift[1])

        # Sovrapponi il layer sull'immagine di output
        output_img = Image.alpha_composite(output_img, layer)

    # Salva l'immagine risultante
    output_img.save(output_path, "PNG")

if __name__ == "__main__":
    # Parametri dello script
    color_image_path = "rgb_000000.jpg"  # Percorso dell'immagine a colori
    depthmap_path = "depth_000000.jpg"  # Percorso della depthmap
    output_path = "displaced_000000.png"  # Percorso di output
    N = 10  # Numero di livelli
    direction = "right"  # Direzione dello shift, "right" o "left"

    # Esegui la funzione
    create_displaced_image(color_image_path, depthmap_path, output_path, N, direction)
