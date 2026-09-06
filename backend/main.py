from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ContextShield API",
    description="Backend API for ContextShield - Social Media Post Safety Checker",
    version="0.1.0",
)

# Enable CORS so the Next.js frontend (running on http://localhost:3000) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Basic health check endpoint returning system status."""
    return {"status": "ok"}
