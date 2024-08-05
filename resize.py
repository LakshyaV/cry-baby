import os
from PIL import Image, ImageOps
import numpy as np

specs_dir = "./mel_specs/"
resized_specs_dir = "./resized_mel_specs/"
img_height, img_width = 128, 256

def pad_specs(image_path, save_path, target_height, target_width):
    with Image.open(image_path) as img:
        width, height = img.size
        left = (target_width - width) // 2
        top = (target_height - height) // 2
        right = target_width - width - left
        bottom = target_height - height - top

        img_padded = ImageOps.expand(img, (left, top, right, bottom), fill='black')
        path = os.path.join(save_path, os.path.basename(image_path))
        img.save(path)

if __name__ == "__main__":
    for file in os.listdir(specs_dir)[1:]:
        for spec_file in os.listdir(os.path.join(specs_dir, file)):
            if spec_file.endswith(".png"):
                new_dir = os.path.join(resized_specs_dir, file)
                spec_dir = os.path.join(specs_dir, file, spec_file)
                if not os.path.exists(os.path.join(resized_specs_dir, file)):
                    os.makedirs(os.path.join(resized_specs_dir, file))
                pad_specs(spec_dir, new_dir, img_height, img_width)