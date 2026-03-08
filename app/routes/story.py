#C:\Users\DELL\Documents\AI_PM_Projects\chromasync\chromasync-api\app\routes\story.py

import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─── Request / Response models ────────────────────────────────────────────────

class LoglineRequest(BaseModel):
    raw_idea: str
    format: str  # "film" or "short_story"

class LoglineVersion(BaseModel):
    label: str
    logline: str
    angle: str

class LoglineResponse(BaseModel):
    versions: list[LoglineVersion]
    primal_question: str

class CharacterRequest(BaseModel):
    logline: str
    format: str
    wound_answer: str
    character_name: str | None = None

class SaveTheCatOption(BaseModel):
    option: str
    scene: str
    framing: str

class CharacterResponse(BaseModel):
    lie: str
    want: str
    need: str
    save_the_cat: list[SaveTheCatOption]
    secondary_character_prompt: str

class BeatRequest(BaseModel):
    beat_number: int
    beat_name: str
    format: str
    logline: str
    character_lie: str
    character_want: str
    character_need: str
    completed_beats: list[dict]

class BeatResponse(BaseModel):
    question: str
    hint: str
    emotional_note: str

# ─── /logline ─────────────────────────────────────────────────────────────────

@router.post("/logline", response_model=LoglineResponse)
async def generate_loglines(req: LoglineRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)

    format_context = (
        "This is a FILM. Use cinematic language appropriate for a screenplay pitch."
        if req.format == "film"
        else "This is a SHORT STORY (1,000-5,000 words). Use literary fiction language."
    )

    prompt = f"""You are a story development expert trained in Blake Snyder's Save the Cat logline principles, Jeff Lyons' Premise Line framework, and Robert McKee's story theory.

{format_context}

A writer has given you their raw story idea:
\"{req.raw_idea}\"

Return THREE logline versions, each emphasising a different angle. Each logline must:
- Contain an ironic or compelling premise
- Name or imply the protagonist
- State a clear goal and clear stakes
- Be one sentence, under 40 words

Then ask the single most important Primal Question to find the deeper emotional truth.

Respond ONLY with valid JSON, no markdown:
{{
  "versions": [
    {{"label": "External Stakes", "logline": "logline focused on external plot danger", "angle": "one sentence on what this emphasises"}},
    {{"label": "Internal Stakes", "logline": "logline focused on internal emotional journey", "angle": "one sentence on what this emphasises"}},
    {{"label": "Tonal Shift", "logline": "logline reframing from unexpected tonal angle", "angle": "one sentence on what this emphasises"}}
  ],
  "primal_question": "a single deep question about what the protagonist desperately wants at their most human level — specific to this story"
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return LoglineResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── Beat definitions ────────────────────────────────────────────────────────────────

FILM_BEATS = [
    {"number": 1,  "name": "Opening Image",         "description": "A single image capturing the hero's world before the adventure begins."},
    {"number": 2,  "name": "Theme Stated",           "description": "Someone states the theme — the lesson the hero must learn."},
    {"number": 3,  "name": "Set-Up",                 "description": "The world, the problem, the hero's flaw and potential for change."},
    {"number": 4,  "name": "Catalyst",               "description": "The inciting incident that upends the hero's world and forces a choice."},
    {"number": 5,  "name": "Debate",                 "description": "The hero resists the call. Fear holds them back."},
    {"number": 6,  "name": "Break into Two",         "description": "The hero makes an active choice and steps into the upside-down world."},
    {"number": 7,  "name": "B Story",                "description": "A new relationship arrives to carry the theme."},
    {"number": 8,  "name": "Fun and Games",          "description": "The promise of the premise. The hero tries to get what they want."},
    {"number": 9,  "name": "Midpoint",               "description": "A false victory or false defeat that raises the stakes."},
    {"number": 10, "name": "Bad Guys Close In",      "description": "External pressure and internal doubt conspire."},
    {"number": 11, "name": "All Is Lost",            "description": "The lowest point. The hero loses everything."},
    {"number": 12, "name": "Dark Night of the Soul",  "description": "The hero wallows. Then — a breakthrough from within."},
    {"number": 13, "name": "Break into Three",       "description": "Armed with new insight, the hero acts."},
    {"number": 14, "name": "Finale",                 "description": "The hero storms the castle and proves the theme."},
    {"number": 15, "name": "Final Image",            "description": "A mirror of the Opening Image showing how the world has changed."},
]

SHORT_BEATS = [
    {"number": 1, "name": "Inciting Moment", "description": "The disruption that breaks the protagonist's equilibrium."},
    {"number": 2, "name": "Rising Pressure", "description": "Tension compounds. The protagonist tries and fails."},
    {"number": 3, "name": "Crisis Point",    "description": "The moment of no return — a choice must be made."},
    {"number": 4, "name": "Climax",          "description": "The protagonist acts from their deepest truth."},
    {"number": 5, "name": "Resonance",       "description": "The final image that carries the emotional aftershock."},
]


# ─── /beat ───────────────────────────────────────────────────────────────────

@router.post("/beat", response_model=BeatResponse)
async def generate_beat_question(req: BeatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    beat_list = FILM_BEATS if req.format == "film" else SHORT_BEATS
    total = len(beat_list)

    completed_summary = ""
    if req.completed_beats:
        lines = [f"Beat {b['number']} ({b['name']}): {b['answer']}" for b in req.completed_beats]
        completed_summary = "\nBeats already completed:\n" + "\n".join(lines)

    prompt = f"""You are a story structure expert trained in Blake Snyder's Save the Cat beat sheet ({total} beats for this format).

The writer is working on beat {req.beat_number} of {total}: \"{req.beat_name}\"

Story context:
- Logline: \"{req.logline}\"
- The Lie the protagonist believes: \"{req.character_lie}\"
- What they Want: \"{req.character_want}\"
- What they Need: \"{req.character_need}\"{completed_summary}

Your job: ask the single most important question to help the writer discover what happens in THIS beat — specific to their story, not generic.

Also give:
- A one-sentence hint if they get stuck (nudge, not the answer)
- A one-sentence emotional note about what the audience should FEEL at this beat

Respond ONLY with valid JSON, no markdown:
{{
  "question": "the single specific question for this beat",
  "hint": "one sentence nudge if stuck",
  "emotional_note": "what the audience feels at this beat"
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return BeatResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /character ───────────────────────────────────────────────────────────────

@router.post("/character", response_model=CharacterResponse)
async def generate_character(req: CharacterRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)

    format_context = (
        "This is a FILM. Save the Cat moments should be visual and cinematic."
        if req.format == "film"
        else "This is a SHORT STORY. Save the Cat moments should be intimate and character-revealing."
    )
    name_context = f"The protagonist's name is {req.character_name}." if req.character_name else "The protagonist's name has not been decided yet."

    prompt = f"""You are a character development expert trained in K.M. Weiland's Creating Character Arcs, David Corbett's The Art of Character, and Blake Snyder's Save the Cat.

{format_context}
{name_context}

Logline: \"{req.logline}\"
Protagonist's wound: \"{req.wound_answer}\"

Derive:
1. THE LIE: The false belief the protagonist carries because of this wound. One sentence.
2. WANT vs NEED: Want = external conscious goal. Need = internal truth they must learn. These must tension each other.
3. TWO SAVE THE CAT MOMENTS: Option A = active (protagonist does something), Option B = passive (something happens to them).
4. SECONDARY CHARACTER PROMPT: One question about who resists the protagonist's change and why.

Respond ONLY with valid JSON, no markdown:
{{
  "lie": "the single false belief",
  "want": "the external conscious goal",
  "need": "the internal truth they must learn",
  "save_the_cat": [
    {{"option": "A", "scene": "vivid 2-3 sentence scene description", "framing": "active"}},
    {{"option": "B", "scene": "vivid 2-3 sentence scene description", "framing": "passive"}}
  ],
  "secondary_character_prompt": "specific question about who resists this protagonist's change"
}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return CharacterResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
