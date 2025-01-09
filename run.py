import os
import glob
import torch
import utils
import cv2
import time
import numpy as np
from midas.model_loader import load_model

def process(device, model, image, input_size, target_size):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    prediction = model.forward(sample)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=target_size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return prediction


def run(input_path, output_path, model_type="dpt_beit_large_512"):
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load model using the default model for the given model type
    model, transform, net_w, net_h = load_model(device, None, model_type)

    # get input
    image_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(image_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("Start processing")
    start_time = time.time()

    for index, image_name in enumerate(image_names):
        print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

        # input
        original_image_rgb = utils.read_image(image_name)  # in [0, 1]
        image = transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = process(device, model, image, (net_w, net_h), original_image_rgb.shape[1::-1])

        # output
        filename = os.path.join(output_path, os.path.splitext(os.path.basename(image_name))[0])
        utils.write_depth(filename, prediction, grayscale=True, bits=2)

        # Estimate remaining time
        elapsed_time = time.time() - start_time
        images_processed = index + 1
        images_remaining = num_images - images_processed
        avg_time_per_image = elapsed_time / images_processed if images_processed > 0 else 0
        estimated_time_remaining = avg_time_per_image * images_remaining

        elapsed_time_str = "{:.0f}s".format(elapsed_time) if elapsed_time < 60 else "{:.0f}m".format(elapsed_time / 60)
        time_remaining_str = "{:.0f}s".format(estimated_time_remaining) if estimated_time_remaining < 60 else "{:.0f}m".format(estimated_time_remaining / 60)
        print("    Time elapsed: {} | Estimated time remaining: {}".format(elapsed_time_str, time_remaining_str))

    print("Finished")


if __name__ == "__main__":
    input_path = "/content/rgb"  # path to input images
    output_path = "/content/depth"  # path to save depth maps
    model_type = "dpt_beit_large_512"  # default model type

    run(input_path, output_path, model_type)
