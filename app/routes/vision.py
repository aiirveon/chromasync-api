import os
import anthropic
import base64
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter()

class SceneAnalysis(BaseModel):
    shot_type: str
    lighting_feel: str
    colour_mood: str
    suggested_look: str
    technical_note: str
    camera_advice: str

@router.post("/analyse", response_model=SceneAnalysis)
async def analyse_scene(
    file: UploadFile = File(...),
    camera_name: str = "Unknown Camera"
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Vision AI not configured")

    image_data = await file.read()
    b64 = base64.standard_b64encode(image_data).decode("utf-8")
    media_type = file.content_type or "image/jpeg"
    if media_type not in ["image/jpeg","image/png","image/webp","image/gif"]:
        media_type = "image/jpeg"

    client = anthropic.Anthropic(api_key=api_key)
    camera_context = f"The filmmaker is shooting on a {camera_name}." if camera_name != "Unknown Camera" else ""

    prompt = f"""You are a cinematography assistant helping beginner indie filmmakers. Analyse this reference image in simple plain English that a beginner can understand. No jargon.

{camera_context}

Respond ONLY with a JSON object, no markdown, no other text:
{{
  "shot_type": "What kind of shot this looks like e.g. A close-up portrait shot",
  "lighting_feel": "What the lighting feels like e.g. Warm golden light like late afternoon sun through a window",
  "colour_mood": "The overall colour mood e.g. Warm and cosy with orange and brown tones",
  "suggested_look": "A style suggestion e.g. A warm film look like an old home video",
  "technical_note": "One simple observation e.g. The image is quite dark you may need more light",
  "camera_advice": "One plain English tip for achieving this look{' on the ' + camera_name if camera_name != 'Unknown Camera' else ''}"
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": prompt}
        ]}],
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return SceneAnalysis(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")