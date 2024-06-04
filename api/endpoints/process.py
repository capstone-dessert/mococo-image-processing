import io

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from utils.util import read_image_file, removed_background

router = APIRouter()


@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    image = read_image_file(await file.read())
    background_removed_image = removed_background(image)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    background_removed_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")
