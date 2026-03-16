import subprocess
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import pre_shoot, on_shoot, post_correction, vision, compare, story, drift

MODEL_PATH = os.path.join(os.path.dirname(__file__), "app", "models", "colour_correction_model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train model at startup if not present
    if not os.path.exists(MODEL_PATH):
        print("[ChromaSync] Colour correction model not found. Training now...")
        try:
            train_script = os.path.join(os.path.dirname(__file__), "train_model.py")
            subprocess.run([sys.executable, train_script], check=True)
            print("[ChromaSync] Model training complete.")
        except Exception as e:
            print(f"[ChromaSync] Model training failed: {e}. Fallback delta corrections will be used.")
    else:
        print("[ChromaSync] Colour correction model found.")
    yield


app = FastAPI(
    title="ChromaSync API",
    description="AI colour intelligence backend for indie filmmakers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pre_shoot.router, prefix="/api/pre-shoot", tags=["Pre-Shoot"])
app.include_router(on_shoot.router, prefix="/api/on-shoot", tags=["On-Shoot"])
app.include_router(compare.router, prefix="/api/on-shoot", tags=["On-Shoot"])
app.include_router(post_correction.router, prefix="/api/post-correction", tags=["Post Correction"])
app.include_router(vision.router, prefix="/api/vision", tags=["Vision AI"])
app.include_router(story.router, prefix="/api/story", tags=["Story"])
app.include_router(drift.router, prefix="/api/colour", tags=["Colour"])

@app.get("/")
def root():
    return {"status": "ChromaSync API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ping")
def ping():
    return {"pong": True}
