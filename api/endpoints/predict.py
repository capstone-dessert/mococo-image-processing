from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from models.model import load_model, predict
from utils.util import read_image_file, preprocess_image, get_decoded_target, removed_background, get_color

router = APIRouter()


class PredictResponse(BaseModel):
    category: str
    color: str


model = load_model("data/2024-05-07-1906_model_epoch_30.pth", num_classes=5)


@router.post("/predict", response_model=PredictResponse)
async def get_predictions(file: UploadFile = File(...)):
    image = read_image_file(await file.read())
    predictions = predict(model, preprocess_image(image))

    removed_background_image = removed_background(image)
    color = get_color(removed_background_image)
    return PredictResponse(category=get_decoded_target(predictions), color=color)
