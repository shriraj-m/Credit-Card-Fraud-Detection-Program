from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API for detecting credit card fraud from datasets using machine learning models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Credit Card Fraud Detection API"
    }


