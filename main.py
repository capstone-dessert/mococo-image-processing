from fastapi import FastAPI

from api.endpoints import predict, process

app = FastAPI()

app.include_router(predict.router)
app.include_router(process.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
