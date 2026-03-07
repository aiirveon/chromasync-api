from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import pre_shoot, on_shoot, post_correction

app = FastAPI(
    title="ChromaSync API",
    description="AI colour intelligence backend for indie filmmakers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://chromasync-app.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pre_shoot.router, prefix="/api/pre-shoot", tags=["Pre-Shoot"])
app.include_router(on_shoot.router, prefix="/api/on-shoot", tags=["On-Shoot"])
app.include_router(post_correction.router, prefix="/api/post-correction", tags=["Post Correction"])

@app.get("/")
def root():
    return {"status": "ChromaSync API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}
