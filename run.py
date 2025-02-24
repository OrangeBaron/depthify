import os
import glob
import torch
import utils
import cv2
import time

from midas.model_loader import load_model, default_models

def process(device, model, image, input_size, target_size, optimize):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    if optimize and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last).half()
    prediction = model.forward(sample)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=target_size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return prediction

def format_time(seconds):
    """Format seconds into hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    time_str = ""
    if hours > 0:
        time_str += f"{hours}h "
    if minutes > 0 or hours > 0:
        time_str += f"{minutes}m "
    time_str += f"{seconds}s"
    return time_str

def run(input_path, output_path, model_type="dpt_beit_large_512", optimize=False, grayscale=False):
    # seleziona il dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # carica il modello predefinito
    model, transform, net_w, net_h = load_model(device, default_models[model_type], model_type, optimize)

    # ottieni le immagini di input
    image_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(image_names)

    # crea la cartella di output
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")
    start_time = time.time()

    for index, image_name in enumerate(image_names):
        print(f"  Processing {image_name} ({index + 1}/{num_images})")

        # input
        original_image_rgb = utils.read_image(image_name)  # in [0, 1]
        image = transform({"image": original_image_rgb})["image"]

        # calcola la depthmap
        with torch.no_grad():
            prediction = process(device, model, image, (net_w, net_h), original_image_rgb.shape[1::-1], optimize)

        # salva il risultato
        if output_path is not None:
            filename = os.path.join(output_path, os.path.splitext(os.path.basename(image_name))[0])
            utils.write_depth(filename, prediction, grayscale, bits=2)
        
        # stima il tempo rimanente
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / (index + 1) if index + 1 > 0 else 0
        images_remaining = num_images - (index + 1)
        estimated_time_remaining = avg_time_per_image * images_remaining
        print(f"    Time elapsed: {format_time(elapsed_time)} | Estimated time remaining: {format_time(estimated_time_remaining)}")

    print("Finished")

if __name__ == "__main__":
    # Parametri hardcoded per il tuo caso d'uso
    input_path = '/content/rgb'
    output_path = '/content/depth'
    
    # Chiamata a run senza bisogno di model_weights
    run(input_path, output_path, model_type="dpt_beit_large_512", optimize=False, grayscale=True)
