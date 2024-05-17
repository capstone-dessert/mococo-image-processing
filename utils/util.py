import io

from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F
from rembg import remove


CLOTHING_CATEGORIES = ["아우터", "하의", "원피스", "상의"]


def read_image_file(file) -> Image.Image:
    return Image.open(io.BytesIO(file))


def preprocess_image(image) -> Tensor:
    return F.to_tensor(image)


def get_decoded_target(prediction: list) -> str:
    return CLOTHING_CATEGORIES[prediction[0]['labels'][0]]


def removed_background(image: Image.Image) -> Image.Image:
    return remove(image)
