import io
import json

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from models.model import load_model, predict
from utils.util import read_image_file, preprocess_image, get_decoded_target, removed_background

router = APIRouter()


class PredictResponse(BaseModel):
    predictions: str


model = load_model("data/2024-05-07-1906_model_epoch_30.pth", num_classes=5)


@router.post("/predict", response_model=PredictResponse)
async def get_predictions(file: UploadFile = File(...)):
    image = read_image_file(await file.read())
    predictions = predict(model, preprocess_image(image))
    background_removed_image = removed_background(image)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    background_removed_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Create a multipart response
    boundary = "boundary"
    response = StreamingResponse(
        iter([f"--{boundary}\r\n".encode('utf-8'),
              'Content-Type: application/json\r\n\r\n',
              json.dumps({"predictions": get_decoded_target(predictions)}, ensure_ascii=False).encode('utf-8'),
              f"\r\n--{boundary}\r\n".encode('utf-8'),
              'Content-Disposition: attachment; filename="image.png"\r\n',
              'Content-Type: image/png\r\n\r\n',
              image_bytes.read(),
              f"\r\n--{boundary}--\r\n".encode('utf-8')]),
        media_type=f"multipart/mixed; boundary={boundary}"
    )

    return response
