from fastapi import FastAPI

from api.endpoints import predict

app = FastAPI()

app.include_router(predict.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
