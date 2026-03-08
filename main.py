from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import pre_shoot, on_shoot, post_correction, vision, compare, story

app = FastAPI(
    title="ChromaSync API",
    description="AI colour intelligence backend for indie filmmakers",
    version="1.0.0"
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

@app.get("/")
def root():
    return {"status": "ChromaSync API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ping")
def ping():
    return {"pong": True}
