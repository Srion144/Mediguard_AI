# main.py — FastAPI app ka entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict

app = FastAPI(title="MediGuard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # baad mein restrict karna
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "MediGuard AI backend chal raha hai ✅"}