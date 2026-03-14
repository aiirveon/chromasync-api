import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)

    framework_ctx = get_framework_context(req.framework, req.format)

    if req.question_number == 1:
        idea_note = """NOTE: If the idea is a title or just a few words, treat it as the seed of the story — not literally.
A title like 'The Wedding Party' does not mean the story is set at a wedding venue.
Think about what kind of world, environment or institution would make THIS story surprising and specific."""
        prompt = f"""A writer has this story idea or title: "{req.raw_idea}"
Framework: {framework_ctx}
{idea_note}

Generate 3 SPECIFIC, SURPRISING location suggestions for where this story could be set.
NOT generic expected settings for the title — subvert expectations.
NOT generic cities or countries. Think: a specific type of building, institution, neighbourhood, environment.
Each suggestion adds texture and unexpected possibility to this idea.
Each suggestion: under 12 words. Vivid and concrete.

{AVOID_LIST}

Respond ONLY with valid JSON, no markdown:
{{"suggestions": ["location 1", "location 2", "location 3"]}}"""

    elif req.question_number == 2:
        ctx2 = ""
        if req.location: ctx2 += f'\nSetting the writer chose: "{req.location}"'
        if req.theme: ctx2 += f'\nTheme: "{req.theme}"'
        prompt = f"""A writer has this story idea or title: "{req.raw_idea}"
{ctx2}
Framework: {framework_ctx}

Generate 3 SPECIFIC broken relationship suggestions that existed BEFORE this story begins.
Not a plot point — something that already happened and left a mark.
Write them as plain story notes — direct and factual, not literary.
Must feel specific to the idea, title and any setting provided. Avoid parent/child estrangement as default.
If the setting was provided, ground the relationship in that world specifically.
Each suggestion: plain language, one sentence, under 20 words.

Examples of the RIGHT tone: "a former partner she informed on, now released from prison", "a cousin who took the money and moved to Abuja"
Examples of the WRONG tone: "She cut off her childhood best friend after discovering she'd been informing" — too literary

{AVOID_LIST}

Respond ONLY with valid JSON, no markdown:
{{"suggestions": ["relationship 1", "relationship 2", "relationship 3"]}}"""

    else:
        ctx3 = ""
        if req.location: ctx3 += f'\nSetting: "{req.location}"'
        if req.broken_relationship: ctx3 += f'\nBroken relationship the writer committed: "{req.broken_relationship}"'
        if req.theme: ctx3 += f'\nTheme: "{req.theme}"'
        prompt = f"""A writer has this story idea or title: "{req.raw_idea}"
{ctx3}
Framework: {framework_ctx}

Generate 3 SPECIFIC private behaviour suggestions — small things the protagonist does when no one is watching.
These must be written as plain, rough story notes — NOT polished prose, NOT literary sentences.
Write them the way a writer would jot them down for themselves: direct, specific, grounded in the details provided.
Not dramatic. Not poetic. Just specific and human.
Must feel directly connected to the setting and broken relationship provided above.
Avoid generic introspective behaviours like journaling, looking in mirrors, or crying alone.
Each suggestion: plain language, under 15 words, starting with the action not the character.

Examples of the RIGHT tone: "counts the exits in every room before sitting down", "keeps a second phone charged with no contacts"
Examples of the WRONG tone: "She rehearses casual lies to the mirror, practicing her innocent face" — too literary

{AVOID_LIST}

Respond ONLY with valid JSON, no markdown:
{{"suggestions": ["behaviour 1", "behaviour 2", "behaviour 3"]}}"""

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
        return InterrogationHintResponse(**json.loads(text.strip()))
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)

    specificity = ""
    if req.location:
        specificity += f"\nSetting: {req.location}"
    if req.broken_relationship:
        specificity += f"\nBroken relationship before the story: {req.broken_relationship}"
    if req.private_behaviour:
        specificity += f"\nThe protagonist when no one is watching: {req.private_behaviour}"

    title_note = "If the idea is a title or just a few words, do NOT treat it literally. A title like 'The Wedding Party' could be about any human conflict that touches on ceremony, obligation, performance, or gathering. Treat the title as a prompt to invent a specific story."

    prompt = f"""You are a story development expert working within this framework:
{framework_ctx}

A writer has this raw idea or title: "{req.raw_idea}"
{specificity}
{title_note}

{AVOID_LIST}

Generate THREE logline versions, each emphasising a different angle.
If specificity details are provided, use them to ground the loglines concretely.
If only a title or short phrase was given, invent three different specific interpretations — each one a genuinely different story.
Use the specificity details above — the setting, broken relationship, and behaviour — to make each logline concrete and grounded in THIS writer's world, not a generic version of their idea.

Each logline must:
- Contain an ironic or surprising premise
- Name or strongly imply the protagonist
- State a clear goal and clear stakes
- Be one sentence, under 40 words
- Feel like it could only be THIS story — not any story

Then ask the single most important Primal Question — the deeper emotional truth beneath the idea.

Respond ONLY with valid JSON, no markdown:
{{
  "versions": [
    {{"label": "External Stakes", "logline": "logline focused on external plot tension", "angle": "one sentence on what this angle emphasises"}},
    {{"label": "Internal Stakes", "logline": "logline focused on internal emotional journey", "angle": "one sentence on what this angle emphasises"}},
    {{"label": "Tonal Shift", "logline": "logline reframing from unexpected angle", "angle": "one sentence on what this angle emphasises"}}
  ],
  "primal_question": "a single deep question specific to this story and its protagonist"
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


# ─── /logline-single ──────────────────────────────────────────────────────────

@router.post("/logline-single", response_model=LoglineSingleResponse)
async def regenerate_single_logline(req: LoglineSingleRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)

    specificity = ""
    if req.location:
        specificity += f"\nSetting: {req.location}"
    if req.broken_relationship:
        specificity += f"\nBroken relationship: {req.broken_relationship}"
    if req.private_behaviour:
        specificity += f"\nProtagonist privately: {req.private_behaviour}"

    existing = ""
    if req.existing_loglines:
        existing = "\nDo NOT produce anything similar to these existing versions:\n" + "\n".join(f"- {l}" for l in req.existing_loglines)

    prompt = f"""You are a story development expert working within this framework:
{framework_ctx}

Raw idea: "{req.raw_idea}"
{specificity}
{existing}

{AVOID_LIST}

Generate ONE new logline for the "{req.label}" angle.
It must be completely different from any existing versions listed above.
Under 40 words. Concrete, specific, surprising.

Respond ONLY with valid JSON, no markdown:
{{"label": "{req.label}", "logline": "the logline", "angle": "one sentence on what this emphasises"}}"""

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
        return LoglineSingleResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")


# ─── /character ───────────────────────────────────────────────────────────────

@router.post("/character", response_model=CharacterResponse)
async def generate_character(req: CharacterRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)
    name_context = f"The protagonist's name is {req.character_name}." if req.character_name else "The protagonist's name has not been decided yet."

    prompt = f"""You are a character development expert trained in K.M. Weiland's Creating Character Arcs and David Corbett's The Art of Character.

Framework: {framework_ctx}
{name_context}

Logline: "{req.logline}"
Protagonist's wound: "{req.wound_answer}"

{AVOID_LIST}

Derive:
1. THE LIE: The specific false belief this protagonist carries because of this wound. Must be unique to their wound — not a generic "I am unlovable" or "I don't deserve happiness."
2. WANT vs NEED: Want = external conscious goal. Need = internal truth they resist. These must create genuine tension with each other.
3. TWO SAVE THE CAT MOMENTS: Specific, visual, surprising. Option A = active (protagonist does something that reveals character). Option B = passive (something happens that reveals character).
4. SECONDARY CHARACTER PROMPT: One question about who is most threatened by this protagonist changing — and why.

Respond ONLY with valid JSON, no markdown:
{{
  "lie": "the specific false belief",
  "want": "the external conscious goal",
  "need": "the internal truth they resist",
  "save_the_cat": [
    {{"option": "A", "scene": "vivid 2-3 sentence scene description", "framing": "active"}},
    {{"option": "B", "scene": "vivid 2-3 sentence scene description", "framing": "passive"}}
  ],
  "secondary_character_prompt": "specific question about who is threatened by this protagonist's change"
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


# ─── /beat ────────────────────────────────────────────────────────────────────

@router.post("/beat", response_model=BeatResponse)
async def generate_beat_question(req: BeatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    beat_list = get_beats(req.format, req.framework)
    total = len(beat_list)
    framework_ctx = get_framework_context(req.framework, req.format)

    completed_summary = ""
    if req.completed_beats:
        lines = [f"Beat {b['number']} ({b['name']}): {b['answer']}" for b in req.completed_beats]
        completed_summary = "\nBeats already completed:\n" + "\n".join(lines)

    prompt = f"""You are a story structure expert working within this framework:
{framework_ctx}

The writer is working on beat {req.beat_number} of {total}: "{req.beat_name}"

Story context:
- Logline: "{req.logline}"
- The Lie the protagonist believes: "{req.character_lie}"
- What they Want: "{req.character_want}"
- What they Need: "{req.character_need}"{completed_summary}

{AVOID_LIST}

Ask the single most important question to help the writer discover what happens in THIS beat — grounded in their specific story, not a generic beat description.
The question should feel surprising and specific — it should be impossible to answer generically.

Also give:
- A one-sentence hint if they get stuck (a nudge toward specificity, not the answer)
- A one-sentence emotional note about what the AUDIENCE should feel at this beat

Respond ONLY with valid JSON, no markdown:
{{
  "question": "the single specific question for this beat",
  "hint": "one sentence nudge toward specificity",
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


# ─── /beat-suggestion ────────────────────────────────────────────────────────

@router.post("/beat-suggestion", response_model=BeatSuggestionResponse)
async def generate_beat_suggestions(req: BeatSuggestionRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    beat_list = get_beats(req.format, req.framework)
    total = len(beat_list)
    framework_ctx = get_framework_context(req.framework, req.format)

    completed_summary = ""
    if req.completed_beats:
        lines = [f"Beat {b['number']} ({b['name']}): {b['answer']}" for b in req.completed_beats]
        completed_summary = "\nBeats already completed:\n" + "\n".join(lines)

    prompt = f"""You are a story structure expert working within this framework:
{framework_ctx}

The writer is working on beat {req.beat_number} of {total}: "{req.beat_name}"

Story context:
- Logline: "{req.logline}"
- The Lie the protagonist believes: "{req.character_lie}"
- What they Want: "{req.character_want}"
- What they Need: "{req.character_need}"{completed_summary}

{AVOID_LIST}

Generate 3 SHORT, SPECIFIC beat answer suggestions for this beat.
Each suggestion should be:
- 1-2 sentences max — a concrete story moment, not a description
- Grounded in THIS specific story's characters, setting, and stakes
- Different from each other in approach or tone
- Written as if the writer wrote it themselves — first draft energy, not polished
- Specific enough to be surprising, not generic enough to fit any story

Respond ONLY with valid JSON, no markdown:
{{"suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]}}"""

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
        return BeatSuggestionResponse(**json.loads(text.strip()))
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
    avoid_current = f'\nDo NOT suggest anything similar to the current theme: "{req.current_theme}"' if req.current_theme else ""

    prompt = f"""You are a story development expert.

Framework: {framework_ctx}
{story_ctx}
{avoid_current}

{AVOID_LIST}

Generate 3 DIFFERENT primal questions — the deeper moral or emotional truth beneath this story.
Each question should:
- Be specific to THIS story, not a generic "what does it mean to love"
- Point toward the protagonist's wound and the lie they believe
- Be a question the story itself is trying to answer, not a plot question
- Be one sentence, phrased as a genuine question

Examples of the RIGHT tone: "Can a person protect someone they love by becoming exactly what they feared?", "Is loyalty to family worth more than loyalty to truth?"
Examples of the WRONG tone: "What is the meaning of love?" — too generic

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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)

    field_instructions = {
        "lie": "THE LIE: The specific false belief this protagonist carries because of their wound. Must be unique to their story — not generic like 'I am unlovable'. One sentence.",
        "want": "WHAT THEY WANT: The external conscious goal the protagonist is chasing. Must create tension with their need. One sentence.",
        "need": "WHAT THEY NEED: The internal truth the protagonist must learn. Must be different from and in tension with what they want. One sentence.",
    }

    story_ctx = ""
    if req.character_name: story_ctx += f"\nProtagonist's name: {req.character_name}"
    if req.location: story_ctx += f"\nSetting: {req.location}"
    if req.broken_relationship: story_ctx += f"\nBroken relationship before story: {req.broken_relationship}"
    if req.private_behaviour: story_ctx += f"\nPrivate behaviour: {req.private_behaviour}"
    if req.theme: story_ctx += f"\nTheme: {req.theme}"
    existing = ""
    if req.current_lie: existing += f"\nCurrent Lie: {req.current_lie}"
    if req.current_want: existing += f"\nCurrent Want: {req.current_want}"
    if req.current_need: existing += f"\nCurrent Need: {req.current_need}"

    prompt = f"""You are a character development expert.

Framework: {framework_ctx}
Logline: "{req.logline}"
Protagonist's wound: "{req.wound_answer}"
{story_ctx}
{existing}

{AVOID_LIST}

Generate a NEW version of: {field_instructions.get(req.field, req.field)}
Must be completely different from the current version shown above.
Specific, surprising, grounded in this protagonist's wound and logline.

Respond ONLY with valid JSON, no markdown:
{{"value": "the new {req.field}"}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return CharacterFieldResponse(**json.loads(text.strip()))
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AI not configured")

    client = anthropic.Anthropic(api_key=api_key)
    framework_ctx = get_framework_context(req.framework, req.format)

    framing_instruction = (
        "ACTIVE: The protagonist does something that reveals their character — they initiate, choose, act."
        if req.framing == "active"
        else "PASSIVE: Something happens to the protagonist that reveals their character — they respond, react, endure."
    )

    avoid_existing = ""
    if req.existing_scene: avoid_existing += f"\nDo NOT produce anything similar to this existing scene: {req.existing_scene}"
    if req.other_scene: avoid_existing += f"\nAlso avoid similarity to: {req.other_scene}"

    prompt = f"""You are a character development expert.

Framework: {framework_ctx}
Logline: "{req.logline}"
Protagonist's wound: "{req.wound_answer}"
The Lie they believe: "{req.lie}"
{avoid_existing}

{AVOID_LIST}

Write ONE new Save the Cat scene for Option {req.option}.
{framing_instruction}
The scene must:
- Be vivid, specific, 2-3 sentences
- Make the audience root for this protagonist before anything goes wrong
- Be grounded in THIS specific story's world and character
- Feel completely different from any existing scene shown above

Respond ONLY with valid JSON, no markdown:
{{"option": "{req.option}", "scene": "the scene description", "framing": "{req.framing}"}}"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=384,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = message.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return SaveTheCatSingleResponse(**json.loads(text.strip()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")