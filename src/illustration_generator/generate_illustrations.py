"""Batch illustration generator using Gemini 3 Pro Image.

Parses a folder of ``day_NN_sharable_image_text.md`` files (each carrying the
verse of the day in English, Tibetan, and Hindi), pulls in the matching daily
"challenge" (practice + explanation) from the Tibetan and English day-plan
files, and submits a batch image generation job with one combined prompt per
day. Generates exactly one illustration per day and saves it with a
transparent background as ``data/illustrations/Day<N>-ch<C>.png``.
"""

import argparse
import io
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from google import genai

from .image_utils import save_png_under_limit

# Default locations of the source vault folders. Overridable via environment
# variables so this script can run on other machines/checkouts.
DALAI_LAMA_PLANS_DIR = Path(os.environ.get(
    "DALAI_LAMA_PLANS_DIR",
    "/Users/tenkal/webuddhist/obsidian/bodhisattvacharyavatara-rails/"
    "3-TRANSFORMATIONS/Plans/Dalai Lama",
))
EN_CHALLENGE_DAYS_DIR = Path(os.environ.get(
    "EN_CHALLENGE_DAYS_DIR",
    "/Users/tenkal/webuddhist/obsidian/bodhisattvacharyavatara-rails/"
    "3-TRANSFORMATIONS/Plans/the-bodhisattva-challenge/en/Days",
))

OUTPUT_DIR = Path(__file__).parent / "data" / "illustrations"

# Raw illustrations are line-art source assets, not the final social-media
# post, so they don't need the strict ~1MB limit used for composed posts.
# Keep a generous cap just to avoid pathological file sizes.
RAW_ILLUSTRATION_MAX_BYTES = 8_000_000  # 8 MB

_STYLE_BLOCK = """\
Create a mature, sophisticated Indian folk-art illustration inspired by the intricate visual language of Madhubani painting and finely observed editorial line art. \
Do not imitate a children's book, cartoon, comic strip, animation, caricature, or mascot style. \
Use natural human proportions, believable anatomy, age-appropriate faces, subtle expressions, and an observational sense of gesture and movement. \
The drawing is exclusively a refined monochrome deep-blue pen line on a plain white or transparent background, with varied line weight, delicate hatching, and carefully controlled detail. \
Use intricate Madhubani-inspired botanical, textile, and architectural patterns selectively as surface detail, not as noisy decoration. \
The scene happens in a rural Indian village in Buddhist Bihar. \
The scene can show the Buddha and bodhisattvas when relevant. \
The composition is one calm, well-observed narrative scene with a clear action and an editorial, museum-quality finish. \
Let the line-art scene float freely with generous breathing room. \
Never draw an outer frame, rectangular boundary, panel outline, mat, picture border, or decorative enclosing line."""

_RULES_BLOCK = """\
STRICT RULES — follow all of them without exception:
- NO text, letters, words, labels, captions, or inscriptions anywhere in the image. \
- The drawing is done exclusively with a 2mm drawing pen line in monochrome blue lines. \
- Don't use snakes of fishes in the trees.
- NO Hindu gods, goddesses, deities, or hindu iconography of any kind \
(no Ganesha, Krishna, Shiva, Durga, Hanuman, or any other deity).
- NO symbols associated with Hinduism. No bindu on people's foreheads.
- The scene must show only ordinary rural people of various ages, animals, nature, or everyday \
village life.
- The illustration should make people smile or gasp.
- The image has one single clear scene, with a clear action happening in the scene.
- ABSOLUTELY NO BORDER: do not draw any line enclosing the artwork on any side. \
No rectangular frame, square box, comic-panel outline, edge rule, border, or decorative perimeter. \
The outermost visible marks must be part of the scene itself, never a continuous line around it.
- Keep the artwork floating on the background; do not make it look like a framed print or tile.
"""

_DAY_VISUAL_BRIEFS = {
    31: """For this day, depict one unmistakable moment at a rural village market: a seller is about to make a quick sale, but openly points out a clearly visible crack or defect in a clay pot instead of hiding it. The buyer looks directly at the disclosed defect and understands the truthful warning; the seller's gesture shows that honesty matters more than the small profit. This exact act of truthful disclosure is the central subject. Do not depict a generic market or a merely pleasant exchange.""",
    67: """For this day, depict one unmistakable moment showing someone catching a flash of anger or craving arising in themselves and visibly halting it before it takes hold, right at that first instant: e.g. a person mid-gesture (a raised hand, a reaching arm, a sharp word half-spoken) who consciously stops themselves, takes a breath, and relaxes, watched by others nearby who notice the shift. The scene must clearly be about intercepting an inner impulse at its very first spark, not about an external danger or threat. Do NOT include any snake, cobra, or other reptile anywhere in the scene, in the trees or on the ground.""",
}


COMBINED_PROMPT_TEMPLATE = f"""\
Let's illustrate a scene for a buddhist text.

Generate exactly one single illustration. Do NOT create multiple scenes or a \
collage — just one clean image.

{_RULES_BLOCK}

Here is today's verse. The Tibetan is the primary text; the English and Hindi \
translations are given only so you can better understand its meaning:

Tibetan verse (primary): {{verse_bo}}
English translation (context only): {{verse_en}}
Hindi translation (context only): {{verse_hi}}

Here is today's practice ("challenge") that goes with the verse, given in \
Tibetan and English for context:

Tibetan practice: {{practice_bo}}
Tibetan explanation: {{explanation_bo}}
English practice: {{practice_en}}
English explanation: {{explanation_en}}

IMPORTANT VISUAL PRIORITY — TODAY'S PRACTICE:
The English practice is the primary subject of the illustration, not merely background context. \
The English explanation is equally important visual guidance: use its concrete examples, motive, consequence, and emotional logic to choose the scene. \
Do not treat the explanation as background reading or replace its specific situation with a generic symbol of honesty. \
Translate the practice and explanation together into one literal, observable human action that could be understood without any text. \
Show the moment of choice and its consequence clearly through body language, gesture, and the objects in the scene. \
Do not make a generic village scene, a generic market scene, or an image that illustrates only an abstract mood. \
Do not illustrate only the verse while ignoring the practice and explanation. \
{{visual_brief}}
Use the verse's themes only to deepen the scene's meaning after the concrete practice and explanation have been established.

{_STYLE_BLOCK}
"""

POLL_INTERVAL_SECONDS = 30

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@dataclass
class DayVerse:
    """The verse of the day, parsed from a sharable-image markdown file."""

    day: int
    chapter: int
    verse: int
    verse_bo: str
    verse_en: str
    verse_hi: str


@dataclass
class DayChallenge:
    """The challenge (practice + explanation) matched to a day's verse."""

    practice_bo: str
    explanation_bo: str
    practice_en: str
    explanation_en: str


def parse_day_number(md_path: Path) -> int:
    """Extract the day number from a filename like ``day_27_...md``."""
    match = re.search(r"day[_-]?(\d{1,3})", md_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find a day number in filename: {md_path.name}")
    return int(match.group(1))


def _detect_script(heading: str) -> str:
    lower = heading.strip().lower()
    if lower.startswith("english"):
        return "english"
    if lower.startswith("tibetan"):
        return "tibetan"
    if lower.startswith("hindi"):
        return "hindi"
    return ""


def _extract_bold_field(text: str, *labels: str) -> str:
    """Extract the text following a ``**label...**`` marker, up to the next ``**``.

    Trailing punctuation inside the bold marker (e.g. ``**ཚིགས་བཅད།**`` or
    ``**आज का श्लोक:**``) is tolerated between the label and the closing ``**``.
    """
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = rf"\*\*\s*(?:{label_pattern})[^\n*]*\*\*[:\s]*\n?(.+?)(?=\n\*\*|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def parse_sharable_image_md(md_path: Path) -> DayVerse:
    """Parse the verse of the day (English/Tibetan/Hindi) from one day file."""
    text = md_path.read_text(encoding="utf-8")
    day = parse_day_number(md_path)

    sections: dict[str, str] = {}
    for heading, body in re.findall(r"(?m)^##\s+(.+?)\s*$\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL):
        script = _detect_script(heading)
        if script:
            sections[script] = body.split("\n---", 1)[0]

    missing_langs = [lang for lang in ("english", "tibetan", "hindi") if lang not in sections]
    if missing_langs:
        raise ValueError(f"{md_path.name}: missing language section(s): {', '.join(missing_langs)}")

    verse_en = _extract_bold_field(sections["english"], "Verse of the day", "Verse of day")
    verse_bo = _extract_bold_field(sections["tibetan"], "ཚིགས་བཅད")
    verse_hi = _extract_bold_field(sections["hindi"], "आज का श्लोक")

    verse_id_raw = _extract_bold_field(sections["english"], "Verse id", "Verse ID")
    id_match = re.search(r"\^(\d+)-(\d+)", verse_id_raw)
    if not id_match:
        raise ValueError(f"{md_path.name}: could not find a chapter-verse id (e.g. ^2-30) in {verse_id_raw!r}")
    chapter, verse = int(id_match.group(1)), int(id_match.group(2))

    if not verse_en or not verse_bo or not verse_hi:
        raise ValueError(f"{md_path.name}: could not parse verse text in all three languages")

    return DayVerse(
        day=day, chapter=chapter, verse=verse,
        verse_bo=verse_bo, verse_en=verse_en, verse_hi=verse_hi,
    )


def _find_chapter_dir(base_dir: Path, chapter: int) -> Path:
    """Find the ``Chapter-<N> ...`` subfolder for a given chapter number."""
    pattern = re.compile(rf"^Chapter-{chapter}\b", re.IGNORECASE)
    matches = [p for p in base_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not matches:
        raise FileNotFoundError(f"No 'Chapter-{chapter}' folder found under {base_dir}")
    return matches[0]


def _extract_heading_block(text: str, needle: str, level: str = "###") -> str:
    """Extract the body under a heading (at ``level``) whose text contains ``needle``."""
    pattern = rf"(?m)^{level}[^\n]*{re.escape(needle)}[^\n]*\n(.*?)(?=\n{level}[^#]|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""


def _extract_heading_block_any_level(text: str, needle: str, levels: tuple[str, ...]) -> str:
    """Try ``_extract_heading_block`` at each heading level in turn.

    The day-plan files are hand-authored and the heading level used for a
    given section (e.g. "today's practice") has varied over time (``#``,
    ``###``, ...), so probe several levels instead of assuming a fixed one.
    """
    for level in levels:
        block = _extract_heading_block(text, needle, level=level)
        if block:
            return block
    return ""


def find_tibetan_day_plan(day: int, chapter: int) -> Path:
    """Locate the Tibetan (Dalai Lama) day-plan file for ``day``/``chapter``."""
    chapter_dir = _find_chapter_dir(DALAI_LAMA_PLANS_DIR, chapter)
    pattern = re.compile(rf"^Day-{day}-Ch{chapter}-V[\d-]+\.md$", re.IGNORECASE)
    matches = [p for p in chapter_dir.iterdir() if p.is_file() and pattern.match(p.name)]
    if not matches:
        raise FileNotFoundError(f"No Tibetan day-plan file for day {day} (chapter {chapter}) in {chapter_dir}")
    return matches[0]


def find_english_day_plan(day: int, chapter: int) -> Path:
    """Locate the English day-plan file for ``day``/``chapter``."""
    chapter_dir = _find_chapter_dir(EN_CHALLENGE_DAYS_DIR, chapter)
    pattern = re.compile(rf"^{day}-ch{chapter}-v[\d-]+-eng\.md$", re.IGNORECASE)
    matches = [p for p in chapter_dir.iterdir() if p.is_file() and pattern.match(p.name)]
    if not matches:
        raise FileNotFoundError(f"No English day-plan file for day {day} (chapter {chapter}) in {chapter_dir}")
    return matches[0]


def _extract_field(text: str, needle: str) -> str:
    """Extract a field that may appear either as a heading (``##``/``###``/``####``)
    or as an inline ``**needle**`` bold marker (all conventions occur across
    the Tibetan day-plan files).
    """
    heading_value = _extract_heading_block_any_level(text, needle, levels=("##", "###", "####")).strip()
    if heading_value:
        return heading_value
    return _extract_bold_field(text, needle)


def parse_tibetan_day_plan(md_path: Path, expected_chapter: int, expected_verse: int) -> tuple[str, str]:
    """Extract the Tibetan practice + explanation for today from a day-plan file."""
    text = md_path.read_text(encoding="utf-8")

    section = _extract_heading_block_any_level(text, "དེ་རིང་གི་ཉམས་ལེན", levels=("#", "##", "###"))
    if not section:
        raise ValueError(f"{md_path.name}: could not find the 'today's practice' section")

    practice_bo = _extract_field(section, "ཉམས་ལེན་དངོས")
    explanation_bo = _extract_field(section, "འགྲེལ་བཤད")
    image_verse_block = _extract_field(section, "པར་གྱི་ཚིགས་བཅད")

    if not practice_bo or not explanation_bo:
        raise ValueError(f"{md_path.name}: could not parse Tibetan practice/explanation")

    id_match = re.search(r"\^(\d+)-(\d+)", image_verse_block)
    if id_match:
        chapter, verse = int(id_match.group(1)), int(id_match.group(2))
        if (chapter, verse) != (expected_chapter, expected_verse):
            print(
                f"  WARNING: {md_path.name} 'verse for the image' is "
                f"^{chapter}-{verse}, expected ^{expected_chapter}-{expected_verse}"
            )

    return practice_bo, explanation_bo


def parse_english_day_plan(md_path: Path) -> tuple[str, str]:
    """Extract the English practice + explanation for today from a day-plan file."""
    text = md_path.read_text(encoding="utf-8")

    section = _extract_heading_block(text, "3) Today's Practice", level="##")
    if not section:
        raise ValueError(f"{md_path.name}: could not find the \"Today's Practice\" section")

    practice_en = _extract_bold_field(section, "Actual Practice")
    explanation_en = _extract_bold_field(section, "Explanation")

    if not practice_en or not explanation_en:
        raise ValueError(f"{md_path.name}: could not parse English practice/explanation")

    return practice_en, explanation_en


def load_day_challenge(day_verse: DayVerse) -> DayChallenge:
    """Locate and parse the Tibetan + English day-plan files for one verse."""
    tibetan_path = find_tibetan_day_plan(day_verse.day, day_verse.chapter)
    practice_bo, explanation_bo = parse_tibetan_day_plan(
        tibetan_path, day_verse.chapter, day_verse.verse
    )

    english_path = find_english_day_plan(day_verse.day, day_verse.chapter)
    practice_en, explanation_en = parse_english_day_plan(english_path)

    return DayChallenge(
        practice_bo=practice_bo,
        explanation_bo=explanation_bo,
        practice_en=practice_en,
        explanation_en=explanation_en,
    )


def is_processed(output_path: Path) -> bool:
    """A day is processed if its illustration file already exists."""
    return output_path.exists()


def make_background_transparent(
    image_path: Path,
    output_path: Path | None = None,
    threshold: int = 240,
) -> Path:
    """Replace near-white background pixels with transparency.

    Args:
        image_path: Path to the source PNG image.
        output_path: Where to save the result. Defaults to overwriting image_path.
        threshold: RGB channel minimum to consider a pixel as background (0-255).

    Returns:
        Path to the saved transparent image.
    """
    output_path = output_path or image_path
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    rgb = data[:, :, :3]
    is_background = np.all(rgb > threshold, axis=2)
    data[is_background, 3] = 0

    Image.fromarray(data).save(output_path, "PNG")
    return output_path


def _make_request(prompt: str) -> dict:
    """Wrap a prompt string into a batch request entry."""
    return {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
        "config": {
            "response_modalities": ["TEXT", "IMAGE"],
            "tools": [{"google_search": {}}],
            "image_config": {
                "aspect_ratio": "3:2",
                "image_size": "2K",
            },
        },
    }


def build_request_for_day(day_verse: DayVerse, challenge: DayChallenge) -> dict:
    """Build the single combined batch request for one day's verse + challenge."""
    prompt = COMBINED_PROMPT_TEMPLATE.format(
        verse_bo=day_verse.verse_bo,
        verse_en=day_verse.verse_en,
        verse_hi=day_verse.verse_hi,
        practice_bo=challenge.practice_bo,
        explanation_bo=challenge.explanation_bo,
        practice_en=challenge.practice_en,
        explanation_en=challenge.explanation_en,
        visual_brief=_DAY_VISUAL_BRIEFS.get(
            day_verse.day,
            "Create one concrete, observable scene that directly enacts the English practice above."
        ),
    )
    return _make_request(prompt)


def submit_batch(client: genai.Client, requests: list[dict]) -> str:
    """Submit inline batch job and return the job name."""
    batch_job = client.batches.create(
        model="gemini-3-pro-image-preview",
        src=requests,
        config={"display_name": "challenge-illustrations"},
    )
    print(f"Batch job created: {batch_job.name}")
    return batch_job.name


def poll_until_done(client: genai.Client, job_name: str):
    """Poll batch job until it reaches a terminal state."""
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)
        print(f"  Status: {state}")

        if state in COMPLETED_STATES:
            return batch_job

        time.sleep(POLL_INTERVAL_SECONDS)


def save_images(batch_job, output_paths: list[Path]) -> None:
    """Extract images from batch responses and save one per day.

    Responses are ordered 1:1 with ``output_paths``.
    """
    responses = batch_job.dest.inlined_responses

    for output_path, inline_response in zip(output_paths, responses):
        if inline_response.error:
            print(f"  ERROR for {output_path.name}: {inline_response.error}")
            continue

        if not inline_response.response:
            print(f"  No response for {output_path.name}")
            continue

        for part in inline_response.response.candidates[0].content.parts:
            if part.inline_data:
                # part.as_image() returns a genai Image (raw bytes wrapper),
                # not a PIL Image — decode it before using PIL-based helpers.
                pil_image = Image.open(io.BytesIO(part.as_image().image_bytes))
                # Raw illustrations are intermediate line-art assets, not the
                # final social-media post — the 1MB default in
                # save_png_under_limit forces 256-color palette quantization,
                # which visibly pixelates the anti-aliased line edges. Use a
                # much larger cap so full-quality PNG is kept whenever
                # possible; downstream composition steps apply their own
                # (appropriately strict) size limit to the final post.
                save_png_under_limit(pil_image, output_path, max_bytes=RAW_ILLUSTRATION_MAX_BYTES)
                make_background_transparent(output_path)
                print(f"  Saved: {output_path}")
            elif part.text:
                text_path = output_path.with_suffix(".notes.txt")
                with text_path.open("a", encoding="utf-8") as f:
                    f.write(f"{part.text}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate daily Bodhisattva Challenge illustrations via Gemini."
    )
    parser.add_argument("sharable_images_dir", help="Path to the Sharable-Images folder")
    parser.add_argument(
        "--day", type=int, default=None,
        help="Only generate the illustration for this day number (default: all unprocessed days)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate illustrations even when output files already exist",
    )
    parsed = parser.parse_args()
    day_filter = parsed.day

    sharable_dir = Path(parsed.sharable_images_dir).resolve()
    if not sharable_dir.is_dir():
        print(f"Error: {sharable_dir} is not a directory")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        print("Error: GEMINI_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        sharable_dir.glob("day_*_sharable_image_text.md"),
        key=parse_day_number,
    )
    if not md_files:
        print(f"No day_*_sharable_image_text.md files found in {sharable_dir}")
        sys.exit(1)

    if day_filter is not None:
        md_files = [p for p in md_files if parse_day_number(p) == day_filter]
        if not md_files:
            print(f"No day file found for day {day_filter} in {sharable_dir}")
            sys.exit(1)

    print(f"Found {len(md_files)} day file(s) in {sharable_dir.name}")

    requests: list[dict] = []
    output_paths: list[Path] = []

    for md_path in md_files:
        try:
            day_verse = parse_sharable_image_md(md_path)
        except ValueError as exc:
            print(f"  Skipping {md_path.name}: {exc}")
            continue

        output_path = OUTPUT_DIR / f"Day{day_verse.day}-ch{day_verse.chapter}.png"
        if is_processed(output_path) and not parsed.force:
            print(f"  Skipping day {day_verse.day} (already has an illustration)")
            continue

        try:
            challenge = load_day_challenge(day_verse)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  Skipping day {day_verse.day}: {exc}")
            continue

        print(f"  Day {day_verse.day} (ch{day_verse.chapter}, v{day_verse.verse}): queued")
        requests.append(build_request_for_day(day_verse, challenge))
        output_paths.append(output_path)

    if not requests:
        print("Nothing to do.")
        return

    print(f"\nSubmitting batch: {len(output_paths)} day(s)...")
    job_name = submit_batch(client, requests)

    print("Polling for completion...")
    batch_job = poll_until_done(client, job_name)

    state = batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)
    if state == "JOB_STATE_SUCCEEDED":
        print("\nBatch succeeded! Saving images...")
        save_images(batch_job, output_paths)
        print("\nDone.")
    else:
        print(f"\nBatch job ended with state: {state}")
        if hasattr(batch_job, "error") and batch_job.error:
            print(f"Error: {batch_job.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
