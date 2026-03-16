import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─── Shared client — created once at module load, not per request ─────────────
# Avoids recreating the Anthropic client on every API call.
_client: anthropic.Anthropic | None = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="AI not configured")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ─── Prompt caching helpers ───────────────────────────────────────────────────
# The AVOID_LIST is injected into a cached system block on every call.
# Anthropic caches it for 5 minutes — subsequent calls pay only 10% of the
# normal input token cost for those tokens (90% saving).
# The minimum cacheable block size is 1024 tokens — AVOID_LIST qualifies.

def cached_system_block(extra: str = "") -> list:
    """Return the system block with cache_control on the AVOID_LIST."""
    content = AVOID_LIST
    if extra:
        content = extra + "\n\n" + AVOID_LIST
    return [
        {
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def call_claude(
    user_prompt: str,
    max_tokens: int,
    system_extra: str = "",
    use_thinking: bool = False,
    thinking_budget: int = 8000,
) -> str:
    """
    Central Claude call function.

    Prompt caching: the AVOID_LIST is always in a cached system block.
    Extended thinking: enabled only when use_thinking=True (character and beat
    endpoints). Adds a thinking budget so Claude reasons before responding.
    Returns the final text response as a string.
    """
    client = get_client()

    kwargs = dict(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        system=cached_system_block(system_extra),
        messages=[{"role": "user", "content": user_prompt}],
    )

    if use_thinking:
        # Extended thinking — Claude reasons silently before producing the
        # response. The thinking tokens are charged but not returned to the user.
        # Budget must be less than max_tokens.
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # When thinking is enabled, temperature must be 1 (API requirement)
        kwargs["temperature"] = 1

    message = client.messages.create(**kwargs)

    # Extract text from response — skip thinking blocks if present
    for block in message.content:
        if block.type == "text":
            return block.text.strip()

    raise HTTPException(status_code=500, detail="No text response from AI")


def parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON response from Claude."""
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

# ─── Option 4: Negative constraints ──────────────────────────────────────────
# Injected silently into every generation prompt.
# Forces the model off its most common defaults.

AVOID_LIST = """
CRITICAL — AVOID ALL OF THE FOLLOWING. These are the most overused AI story defaults:

OVERUSED CHARACTER WOUNDS (never use these as the central wound):
- Absent or dead parent as the defining trauma
- Childhood abandonment or orphan backstory  
- Survivor's guilt from a death the protagonist caused or witnessed
- Estranged sibling or family member who needs reconciling
- Workaholic who neglected their family and must learn balance
- Soldier/first responder haunted by one specific incident
- Gifted person who was told they'd never amount to anything

OVERUSED STORY STRUCTURES (never default to these):
- Chosen one who must accept their destiny
- Underdog who wins the big competition through heart not talent
- Loner who learns to trust and love again through one transformative relationship
- City person who moves to small town and discovers what really matters
- Person who gets everything they thought they wanted and discovers it's hollow

OVERUSED RESOLUTION SHAPES (never end this way by default):
- The protagonist gives a speech that changes everything
- A misunderstanding is resolved through honest conversation at the last moment
- The protagonist sacrifices themselves and is rewarded for it

Instead: find the specific, surprising, human truth inside THIS writer's idea.
"""

# ─── Option 2: Framework beat lists ──────────────────────────────────────────

SAVE_THE_CAT_BEATS = [
    {"number": 1,  "name": "Opening Image",          "description": "A single image capturing the hero's world before the adventure begins."},
    {"number": 2,  "name": "Theme Stated",            "description": "Someone states the theme — the lesson the hero must learn."},
    {"number": 3,  "name": "Set-Up",                  "description": "The world, the problem, the hero's flaw and potential for change."},
    {"number": 4,  "name": "Catalyst",                "description": "The inciting incident that upends the hero's world and forces a choice."},
    {"number": 5,  "name": "Debate",                  "description": "The hero resists the call. Fear holds them back."},
    {"number": 6,  "name": "Break into Two",          "description": "The hero makes an active choice and steps into the upside-down world."},
    {"number": 7,  "name": "B Story",                 "description": "A new relationship arrives to carry the theme."},
    {"number": 8,  "name": "Fun and Games",           "description": "The promise of the premise. The hero tries to get what they want."},
    {"number": 9,  "name": "Midpoint",                "description": "A false victory or false defeat that raises the stakes."},
    {"number": 10, "name": "Bad Guys Close In",       "description": "External pressure and internal doubt conspire."},
    {"number": 11, "name": "All Is Lost",             "description": "The lowest point. The hero loses everything."},
    {"number": 12, "name": "Dark Night of the Soul",  "description": "The hero wallows. Then — a breakthrough from within."},
    {"number": 13, "name": "Break into Three",        "description": "Armed with new insight, the hero acts."},
    {"number": 14, "name": "Finale",                  "description": "The hero storms the castle and proves the theme."},
    {"number": 15, "name": "Final Image",             "description": "A mirror of the Opening Image showing how the world has changed."},
]

TRUBY_BEATS = [
    {"number": 1,  "name": "Self-Revelation Need & Ghost", "description": "The hero's deep psychological or moral need, and the ghost — the defining past event — that caused it."},
    {"number": 2,  "name": "Problem & Desire",             "description": "The external problem gives the hero a concrete goal. Desire is what they consciously want."},
    {"number": 3,  "name": "Opponent",                     "description": "The opponent is not a villain — they want the same thing and expose the hero's weakness."},
    {"number": 4,  "name": "Plan",                         "description": "The hero's strategy for overcoming the opponent and reaching the goal."},
    {"number": 5,  "name": "Opening Weakness & Need",      "description": "The specific flaw or blindspot the hero carries into the story — moral and psychological."},
    {"number": 6,  "name": "Inciting Event",               "description": "The external event that forces the hero into conflict with the opponent."},
    {"number": 7,  "name": "Desire Established",           "description": "The hero's goal becomes clear and urgent — the audience knows what they're fighting for."},
    {"number": 8,  "name": "Ally & Relationship",          "description": "The ally who helps the hero — and the relationship that will be tested by the moral argument."},
    {"number": 9,  "name": "Moral Argument Begins",        "description": "The story's central moral debate starts — the opponent challenges the hero's worldview."},
    {"number": 10, "name": "First Revelation & Decision",  "description": "The hero learns something that forces a new decision — and reveals a character flaw."},
    {"number": 11, "name": "Gate, Gauntlet & Visit",       "description": "The hero faces a sequence of increasingly difficult challenges that test their plan."},
    {"number": 12, "name": "Second Revelation & Decision", "description": "A deeper truth emerges — the hero is forced to change tactics and confront their moral weakness."},
    {"number": 13, "name": "Apparent Defeat",              "description": "The hero appears to have lost everything — the opponent has won."},
    {"number": 14, "name": "Third Revelation & Decision",  "description": "The hero sees the full truth about themselves and makes a final moral decision."},
    {"number": 15, "name": "Moral Self-Revelation",        "description": "The hero understands the true nature of their weakness and what they must change."},
    {"number": 16, "name": "Climax",                       "description": "The hero confronts the opponent using their new moral understanding."},
    {"number": 17, "name": "Moral Decision at Climax",     "description": "The hero must choose between their old self and their new understanding — with consequences."},
    {"number": 18, "name": "New Equilibrium",              "description": "The world after the conflict — changed by the hero's moral decision."},
]

STORY_CIRCLE_BEATS = [
    {"number": 1, "name": "You",           "description": "Establish the protagonist in their zone of comfort — who they are and what their world looks like."},
    {"number": 2, "name": "Need",          "description": "The protagonist wants or needs something — a lack or desire that disrupts their comfort."},
    {"number": 3, "name": "Go",            "description": "The protagonist enters an unfamiliar situation — crosses into an unknown world."},
    {"number": 4, "name": "Search",        "description": "The protagonist adapts to the new world and searches for what they need — encountering obstacles."},
    {"number": 5, "name": "Find",          "description": "The protagonist gets what they thought they wanted — but it costs something."},
    {"number": 6, "name": "Take",          "description": "The protagonist pays the price — there is a heavy cost for what they found."},
    {"number": 7, "name": "Return",        "description": "The protagonist returns to where they began — but changed by the journey."},
    {"number": 8, "name": "Change",        "description": "The protagonist has fundamentally changed — the world is the same but they are not."},
]

SHORT_BEATS = [
    {"number": 1, "name": "Inciting Moment", "description": "The disruption that breaks the protagonist's equilibrium."},
    {"number": 2, "name": "Rising Pressure", "description": "Tension compounds. The protagonist tries and fails."},
    {"number": 3, "name": "Crisis Point",    "description": "The moment of no return — a choice must be made."},
    {"number": 4, "name": "Climax",          "description": "The protagonist acts from their deepest truth."},
    {"number": 5, "name": "Resonance",       "description": "The final image that carries the emotional aftershock."},
]

# Keep legacy name for backward compatibility
FILM_BEATS = SAVE_THE_CAT_BEATS

def get_beats(format: str, framework: str) -> list:
    if format == "short_story":
        return SHORT_BEATS
    if framework == "truby":
        return TRUBY_BEATS
    if framework == "story_circle":
        return STORY_CIRCLE_BEATS
    return SAVE_THE_CAT_BEATS

def get_framework_context(framework: str, format: str) -> str:
    if format == "short_story":
        return "This is a SHORT STORY (1,000–5,000 words). Use literary fiction language and intimate character focus."
    if framework == "truby":
        return """This story uses JOHN TRUBY'S MORAL ARGUMENT framework.
Key principles:
- Every protagonist has a ghost (defining past event) AND a weakness that causes moral failure — they are hurting someone else, not just themselves
- The story is a moral argument: the opponent challenges the hero's worldview and exposes their moral weakness
- The self-revelation must be moral AND psychological — the hero must change how they treat others, not just how they feel
- Avoid redemption arcs that resolve neatly. Truby stories are morally complex and sometimes end in failure."""
    if framework == "story_circle":
        return """This story uses DAN HARMON'S STORY CIRCLE framework.
Key principles:
- The structure is cyclical — the protagonist ends where they began but fundamentally changed
- The 'need' is unconscious — the protagonist doesn't know what they truly need until they've paid the price for what they wanted
- The 'cost' (step 6) must be real and painful — not symbolic
- The 'change' (step 8) must be visible in behaviour, not just attitude
- Works best for contained, intimate stories. Think Fleabag, Russian Doll, short films."""
    return """This story uses BLAKE SNYDER'S SAVE THE CAT beat sheet.
Key principles:
- The logline must have an ironic premise
- The protagonist must be active — they make choices, not receive them
- The theme is stated early and proved at the finale
- Every beat must connect to the protagonist's internal flaw"""

# ─── Request / Response models ────────────────────────────────────────────────

class LoglineRequest(BaseModel):
    raw_idea: str
    format: str
    framework: str = "save_the_cat"
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""

class LoglineVersion(BaseModel):
    label: str
    logline: str
    angle: str

class LoglineResponse(BaseModel):
    versions: list[LoglineVersion]
    primal_question: str

class LoglineSingleRequest(BaseModel):
    raw_idea: str
    format: str
    framework: str = "save_the_cat"
    label: str
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""
    existing_loglines: list[str] = []

class LoglineSingleResponse(BaseModel):
    label: str
    logline: str
    angle: str

class InterrogationHintRequest(BaseModel):
    question_number: int  # 1, 2, or 3
    raw_idea: str
    format: str
    framework: str = "save_the_cat"
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""
    theme: str = ""  # primal question if already set

class InterrogationHintResponse(BaseModel):
    suggestions: list[str]

class CharacterRequest(BaseModel):
    logline: str
    format: str
    framework: str = "save_the_cat"
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
    framework: str = "save_the_cat"
    logline: str
    character_lie: str
    character_want: str
    character_need: str
    completed_beats: list[dict]

class BeatResponse(BaseModel):
    question: str
    hint: str
    emotional_note: str

class BeatSuggestionRequest(BaseModel):
    beat_number: int
    beat_name: str
    format: str
    framework: str = "save_the_cat"
    logline: str
    character_lie: str
    character_want: str
    character_need: str
    completed_beats: list[dict] = []

class BeatSuggestionResponse(BaseModel):
    suggestions: list[str]


# ─── /interrogation-hints ─────────────────────────────────────────────────────

@router.post("/interrogation-hints", response_model=InterrogationHintResponse)
async def generate_interrogation_hints(req: InterrogationHintRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    if req.question_number == 1:
        prompt = f"""A writer has this story idea or title: "{req.raw_idea}"
Framework: {framework_ctx}
If the idea is a title, treat it as a seed not literally. Think what world would make it surprising.
Generate 3 SPECIFIC surprising location suggestions. Not generic cities. Specific buildings, institutions, environments. Under 12 words each.
Respond ONLY with valid JSON: {{"suggestions": ["location 1", "location 2", "location 3"]}}"""
    elif req.question_number == 2:
        ctx2 = ""
        if req.location: ctx2 += f'\nSetting: "{req.location}"'
        if req.theme: ctx2 += f'\nTheme: "{req.theme}"'
        prompt = f"""Story idea: "{req.raw_idea}"
{ctx2}
Framework: {framework_ctx}
Generate 3 SPECIFIC broken relationships that existed BEFORE the story begins. Plain story notes, direct and factual. One sentence each under 20 words. Grounded in the setting if provided.
Right: "a former partner she informed on, now released from prison"
Wrong: "She cut off her childhood best friend" — too literary
Respond ONLY with valid JSON: {{"suggestions": ["relationship 1", "relationship 2", "relationship 3"]}}"""
    else:
        ctx3 = ""
        if req.location: ctx3 += f'\nSetting: "{req.location}"'
        if req.broken_relationship: ctx3 += f'\nBroken relationship: "{req.broken_relationship}"'
        if req.theme: ctx3 += f'\nTheme: "{req.theme}"'
        prompt = f"""Story idea: "{req.raw_idea}"
{ctx3}
Framework: {framework_ctx}
Generate 3 SPECIFIC private behaviours — small things protagonist does when no one watches. Plain rough notes, not polished prose. Under 15 words each, start with the action.
Right: "counts the exits in every room before sitting down"
Wrong: "She rehearses casual lies to the mirror" — too literary
Respond ONLY with valid JSON: {{"suggestions": ["behaviour 1", "behaviour 2", "behaviour 3"]}}"""
    try:
        text = call_claude(prompt, max_tokens=512)
        return InterrogationHintResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /theme-suggestions ───────────────────────────────────────────────────────

class ThemeSuggestionRequest(BaseModel):
    raw_idea: str
    format: str
    framework: str = "save_the_cat"
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""
    existing_loglines: list[str] = []
    current_theme: str = ""

class ThemeSuggestionResponse(BaseModel):
    suggestions: list[str]

@router.post("/theme-suggestions", response_model=ThemeSuggestionResponse)
async def generate_theme_suggestions(req: ThemeSuggestionRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)

    story_ctx = f'Raw idea: "{req.raw_idea}"'
    if req.location: story_ctx += f'\nSetting: "{req.location}"'
    if req.broken_relationship: story_ctx += f'\nBroken relationship: "{req.broken_relationship}"'
    if req.private_behaviour: story_ctx += f'\nPrivate behaviour: "{req.private_behaviour}"'
    if req.existing_loglines: story_ctx += "\nLoglines: " + " / ".join(req.existing_loglines[:3])
    avoid_current = f'\nDo NOT suggest anything similar to: "{req.current_theme}"' if req.current_theme else ""

    prompt = f"""You are a story development expert.

Framework: {framework_ctx}
{story_ctx}
{avoid_current}

{AVOID_LIST}

Generate 3 different primal questions -- the deeper moral or emotional truth beneath this story.
Each question must:
- Be specific to THIS story, not generic
- Point toward the protagonist's wound and lie
- Be the question the story is answering, not a plot question
- One sentence, phrased as a genuine question

Right tone: "Can a person protect someone they love by becoming exactly what they feared?"
Wrong tone: "What is the meaning of love?" -- too generic

Respond ONLY with valid JSON, no markdown:
{{"suggestions": ["theme 1", "theme 2", "theme 3"]}}"""

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
        return ThemeSuggestionResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /logline ─────────────────────────────────────────────────────────────────

@router.post("/logline", response_model=LoglineResponse)
async def generate_loglines(req: LoglineRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    specificity = ""
    if req.location: specificity += f"\nSetting: {req.location}"
    if req.broken_relationship: specificity += f"\nBroken relationship: {req.broken_relationship}"
    if req.private_behaviour: specificity += f"\nProtagonist privately: {req.private_behaviour}"
    prompt = f"""You are a story development expert working within this framework:
{framework_ctx}
Raw idea: "{req.raw_idea}"
{specificity}
If the idea is a title or short phrase, treat it as a seed not a literal brief. Invent a specific story from it.
Generate THREE logline versions each emphasising a different angle. Each must be under 40 words, surprising, specific to THIS story.
Then ask the single Primal Question — the deeper emotional truth beneath the idea.
Respond ONLY with valid JSON:
{{
  "versions": [
    {{"label": "External Stakes", "logline": "...", "angle": "..."}},
    {{"label": "Internal Stakes", "logline": "...", "angle": "..."}},
    {{"label": "Tonal Shift", "logline": "...", "angle": "..."}}
  ],
  "primal_question": "..."
}}"""
    try:
        text = call_claude(prompt, max_tokens=1024)
        return LoglineResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /logline-single ──────────────────────────────────────────────────────────

@router.post("/logline-single", response_model=LoglineSingleResponse)
async def regenerate_single_logline(req: LoglineSingleRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    specificity = ""
    if req.location: specificity += f"\nSetting: {req.location}"
    if req.broken_relationship: specificity += f"\nBroken relationship: {req.broken_relationship}"
    if req.private_behaviour: specificity += f"\nProtagonist privately: {req.private_behaviour}"
    existing = ""
    if req.existing_loglines:
        existing = "\nDo NOT produce anything similar to:\n" + "\n".join(f"- {l}" for l in req.existing_loglines)
    prompt = f"""You are a story development expert.
{framework_ctx}
Raw idea: "{req.raw_idea}"
{specificity}{existing}
Generate ONE new logline for the "{req.label}" angle. Different from existing versions. Under 40 words.
Respond ONLY with valid JSON: {{"label": "{req.label}", "logline": "...", "angle": "..."}}"""
    try:
        text = call_claude(prompt, max_tokens=512)
        return LoglineSingleResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /character ───────────────────────────────────────────────────────────────

@router.post("/character", response_model=CharacterResponse)
async def generate_character(req: CharacterRequest):
    # Extended thinking enabled — character psychology derivation benefits from
    # deeper reasoning. Claude thinks through the wound before deriving Lie/Want/Need.
    framework_ctx = get_framework_context(req.framework, req.format)
    name_context = f"Protagonist: {req.character_name}." if req.character_name else ""
    prompt = f"""You are a character development expert (K.M. Weiland, David Corbett).
Framework: {framework_ctx}
{name_context}
Logline: "{req.logline}"
Wound: "{req.wound_answer}"
Derive:
1. THE LIE: Specific false belief from this wound. Not generic.
2. WANT vs NEED: External goal vs internal truth. Must create genuine tension.
3. TWO SAVE THE CAT MOMENTS: Option A active, Option B passive. Vivid, 2-3 sentences each.
4. SECONDARY CHARACTER PROMPT: Who is most threatened by this protagonist changing?
Respond ONLY with valid JSON:
{{
  "lie": "...",
  "want": "...",
  "need": "...",
  "save_the_cat": [
    {{"option": "A", "scene": "...", "framing": "active"}},
    {{"option": "B", "scene": "...", "framing": "passive"}}
  ],
  "secondary_character_prompt": "..."
}}"""
    try:
        text = call_claude(prompt, max_tokens=5000, use_thinking=True, thinking_budget=3000)
        return CharacterResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /beat ────────────────────────────────────────────────────────────────────

@router.post("/beat", response_model=BeatResponse)
async def generate_beat_question(req: BeatRequest):
    # Extended thinking enabled — beat questions need to understand the full
    # narrative arc before asking the right question for this specific beat.
    beat_list = get_beats(req.format, req.framework)
    total = len(beat_list)
    framework_ctx = get_framework_context(req.framework, req.format)
    completed_summary = ""
    if req.completed_beats:
        lines = [f"Beat {b['number']} ({b['name']}): {b['answer']}" for b in req.completed_beats]
        completed_summary = "\nCompleted beats:\n" + "\n".join(lines)
    prompt = f"""You are a story structure expert.
{framework_ctx}
Beat {req.beat_number} of {total}: "{req.beat_name}"
Logline: "{req.logline}"
Lie: "{req.character_lie}" | Want: "{req.character_want}" | Need: "{req.character_need}"
{completed_summary}
Ask the single most important question for THIS beat — impossible to answer generically.
Also: a one-sentence hint and a one-sentence emotional note for the audience.
Respond ONLY with valid JSON:
{{"question": "...", "hint": "...", "emotional_note": "..."}}"""
    try:
        text = call_claude(prompt, max_tokens=3000, use_thinking=True, thinking_budget=2000)
        return BeatResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /beat-suggestion ────────────────────────────────────────────────────────

@router.post("/beat-suggestion", response_model=BeatSuggestionResponse)
async def generate_beat_suggestions(req: BeatSuggestionRequest):
    beat_list = get_beats(req.format, req.framework)
    total = len(beat_list)
    framework_ctx = get_framework_context(req.framework, req.format)
    completed_summary = ""
    if req.completed_beats:
        lines = [f"Beat {b['number']} ({b['name']}): {b['answer']}" for b in req.completed_beats]
        completed_summary = "\nCompleted beats:\n" + "\n".join(lines)
    prompt = f"""You are a story structure expert.
{framework_ctx}
Beat {req.beat_number} of {total}: "{req.beat_name}"
Logline: "{req.logline}"
Lie: "{req.character_lie}" | Want: "{req.character_want}" | Need: "{req.character_need}"
{completed_summary}
Generate 3 SHORT specific beat answer suggestions. Each 1-2 sentences, concrete story moment, different in tone, first-draft energy, specific to THIS story.
Respond ONLY with valid JSON: {{"suggestions": ["...", "...", "..."]}}"""
    try:
        text = call_claude(prompt, max_tokens=512)
        return BeatSuggestionResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /theme-suggestions ─────────────────────────────────────────────────────

class ThemeSuggestionRequest(BaseModel):
    raw_idea: str
    format: str
    framework: str = "save_the_cat"
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""
    existing_loglines: list[str] = []
    current_theme: str = ""

class ThemeSuggestionResponse(BaseModel):
    suggestions: list[str]

@router.post("/theme-suggestions", response_model=ThemeSuggestionResponse)
async def generate_theme_suggestions(req: ThemeSuggestionRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    story_ctx = f'Raw idea: "{req.raw_idea}"'
    if req.location: story_ctx += f'\nSetting: "{req.location}"'
    if req.broken_relationship: story_ctx += f'\nBroken relationship: "{req.broken_relationship}"'
    if req.private_behaviour: story_ctx += f'\nPrivate behaviour: "{req.private_behaviour}"'
    if req.existing_loglines: story_ctx += "\nLoglines: " + " / ".join(req.existing_loglines[:3])
    avoid_current = f'\nDo NOT suggest anything similar to: "{req.current_theme}"' if req.current_theme else ""
    prompt = f"""You are a story development expert.
Framework: {framework_ctx}
{story_ctx}{avoid_current}
Generate 3 different primal questions — the deeper moral truth beneath this story.
Each must be specific to THIS story, point to the protagonist's wound, one sentence.
Right tone: "Can a person protect someone they love by becoming exactly what they feared?"
Wrong: "What is the meaning of love?" — too generic
Respond ONLY with valid JSON: {{"suggestions": ["theme 1", "theme 2", "theme 3"]}}"""
    try:
        text = call_claude(prompt, max_tokens=512)
        return ThemeSuggestionResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /character-field ────────────────────────────────────────────────────────

class CharacterFieldRequest(BaseModel):
    field: str  # "lie", "want", or "need"
    logline: str
    format: str
    framework: str = "save_the_cat"
    wound_answer: str
    character_name: str = ""
    location: str = ""
    broken_relationship: str = ""
    private_behaviour: str = ""
    theme: str = ""
    current_lie: str = ""
    current_want: str = ""
    current_need: str = ""

class CharacterFieldResponse(BaseModel):
    value: str

@router.post("/character-field", response_model=CharacterFieldResponse)
async def regenerate_character_field(req: CharacterFieldRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    field_instructions = {
        "lie": "THE LIE: Specific false belief from wound. Not generic. One sentence.",
        "want": "WHAT THEY WANT: External goal. Must create tension with need. One sentence.",
        "need": "WHAT THEY NEED: Internal truth they resist. Different from want. One sentence.",
    }
    story_ctx = ""
    if req.character_name: story_ctx += f"\nProtagonist: {req.character_name}"
    if req.location: story_ctx += f"\nSetting: {req.location}"
    if req.broken_relationship: story_ctx += f"\nBroken relationship: {req.broken_relationship}"
    if req.private_behaviour: story_ctx += f"\nPrivate behaviour: {req.private_behaviour}"
    if req.theme: story_ctx += f"\nTheme: {req.theme}"
    existing = ""
    if req.current_lie: existing += f"\nCurrent Lie: {req.current_lie}"
    if req.current_want: existing += f"\nCurrent Want: {req.current_want}"
    if req.current_need: existing += f"\nCurrent Need: {req.current_need}"
    prompt = f"""You are a character development expert.
{framework_ctx}
Logline: "{req.logline}" | Wound: "{req.wound_answer}"
{story_ctx}{existing}
Generate a NEW version of: {field_instructions.get(req.field, req.field)}
Completely different from current version. Specific, grounded in this wound and logline.
Respond ONLY with valid JSON: {{"value": "..."}}"""
    try:
        text = call_claude(prompt, max_tokens=256)
        return CharacterFieldResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /save-the-cat-single ────────────────────────────────────────────────

class SaveTheCatSingleRequest(BaseModel):
    option: str  # "A" or "B"
    framing: str  # "active" or "passive"
    logline: str
    format: str
    framework: str = "save_the_cat"
    wound_answer: str
    lie: str
    existing_scene: str = ""
    other_scene: str = ""

class SaveTheCatSingleResponse(BaseModel):
    option: str
    scene: str
    framing: str

@router.post("/save-the-cat-single", response_model=SaveTheCatSingleResponse)
async def regenerate_save_the_cat(req: SaveTheCatSingleRequest):
    framework_ctx = get_framework_context(req.framework, req.format)
    framing_instruction = (
        "ACTIVE: protagonist initiates, chooses, acts."
        if req.framing == "active"
        else "PASSIVE: something happens to protagonist that reveals character."
    )
    avoid_existing = ""
    if req.existing_scene: avoid_existing += f"\nNot similar to: {req.existing_scene}"
    if req.other_scene: avoid_existing += f"\nNot similar to: {req.other_scene}"
    prompt = f"""You are a character development expert.
{framework_ctx}
Logline: "{req.logline}" | Wound: "{req.wound_answer}" | Lie: "{req.lie}"
{avoid_existing}
Write ONE Save the Cat scene for Option {req.option}. {framing_instruction}
Vivid, specific, 2-3 sentences. Makes audience root for protagonist. Completely different from existing.
Respond ONLY with valid JSON: {{"option": "{req.option}", "scene": "...", "framing": "{req.framing}"}}"""
    try:
        text = call_claude(prompt, max_tokens=384)
        return SaveTheCatSingleResponse(**parse_json_response(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")