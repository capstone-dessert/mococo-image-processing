import io

import webcolors
from PIL import Image
from colorthief import ColorThief
from rembg import remove
from torch import Tensor
from torchvision.transforms import functional as F

CLOTHING_CATEGORIES = ["아우터", "하의", "원피스", "상의"]


def read_image_file(file) -> Image.Image:
    return Image.open(io.BytesIO(file))


def preprocess_image(image) -> Tensor:
    return F.to_tensor(image)


def get_decoded_target(prediction: list) -> str:
    return CLOTHING_CATEGORIES[prediction[0]['labels'][0]]


def removed_background(image: Image.Image) -> Image.Image:
    return remove(image)


def get_color(image: Image.Image) -> str:
    file = io.BytesIO()
    image.save(file, format="PNG")
    file.seek(0)

    color_thief = ColorThief(file)
    dominant_color = color_thief.get_color(quality=1)
    return closest_color(dominant_color)


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.HTML4_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]
