"""Prompt construction for IKEA-Bench evaluation."""

import json
from pathlib import Path

from .utils import _ensure_imports, get_data_dir

# ---------------------------------------------------------------------------
# Text grounding loader
# ---------------------------------------------------------------------------
_descriptions_cache = None


def load_descriptions(data_dir=None):
    """Load step descriptions JSON.

    Parameters
    ----------
    data_dir : str or Path, optional
        Root data directory (the one containing ``step_descriptions.json``
        or ``benchmark/step_descriptions_all.json``).
        Defaults to ``get_data_dir()``.
    """
    global _descriptions_cache
    if _descriptions_cache is None:
        if data_dir is None:
            data_dir = get_data_dir()
        data_dir = Path(data_dir)
        # Support both HF layout (flat) and legacy layout (benchmark/ subfolder)
        candidates = [
            data_dir / "step_descriptions.json",
            data_dir / "step_descriptions_all.json",
            data_dir / "benchmark" / "step_descriptions.json",
            data_dir / "benchmark" / "step_descriptions_all.json",
        ]
        desc_path = None
        for c in candidates:
            if c.exists():
                desc_path = c
                break
        if desc_path is None:
            raise FileNotFoundError(
                f"step_descriptions.json not found in {data_dir}. "
                f"Run setup_data.py first."
            )
        with open(desc_path) as f:
            descs = json.load(f)
        _descriptions_cache = {}
        for d in descs:
            key = (d["product"], d["step_id"])
            _descriptions_cache[key] = d["description"]
    return _descriptions_cache


def get_text_grounding(product, step_id):
    """Get text description for a manual step."""
    descs = load_descriptions()
    desc = descs.get((product, step_id))
    if desc is None:
        return ""
    parts = []
    for field in ["parts", "action", "tools", "spatial", "result",
                   "warnings", "fasteners", "arrow_directions"]:
        if field in desc and desc[field] and desc[field] != "None visible.":
            parts.append(f"  {field.upper()}: {desc[field]}")
    return "Manual step description:\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# System context per question type
# ---------------------------------------------------------------------------
def _build_system_context(qtype, setting):
    """Build task-specific system context that explains the visual inputs
    and their relationships."""
    tg_note = ""
    if setting == "text_grounding":
        tg_note = (
            " Each instruction diagram is accompanied by a text description that describes "
            "the content of that diagram. The text description is a supplementary annotation "
            "of the diagram -- they refer to the same assembly step."
        )

    if setting == "text_only":
        # Text-only: instructions are text descriptions, no diagram images
        contexts = {
            "1a": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames captured from a real-world assembly video, "
                "showing a person physically assembling furniture.\n"
                "2. INSTRUCTION DESCRIPTIONS: Text descriptions of assembly steps from a furniture "
                "manual. Each description explains what parts are involved, how they connect, and "
                "what the step accomplishes. No visual diagrams are provided.\n"
                "Your task is to determine which instruction description corresponds to the action "
                "shown in the video frames."
            ),
            "1b": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video.\n"
                "2. INSTRUCTION DESCRIPTION: A text description of an assembly step, explaining "
                "what parts are involved and what action is performed.\n"
                "Your task is to judge whether the action shown in the video frames matches "
                "the assembly step described in the text."
            ),
            "1c": (
                "You are evaluating a furniture assembly task. You will see two sets of video frames "
                "(Clip A and Clip B) from different recordings of the same furniture product being assembled. "
                "Your task is to determine whether both clips show the same assembly step or different steps."
            ),
            "2a": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video showing "
                "the current state of assembly.\n"
                "2. ALL INSTRUCTION DESCRIPTIONS: The complete set of step descriptions "
                "for this product, in order (Step 1, Step 2, etc.). Each describes what parts "
                "are used and what assembly action is performed.\n"
                "Your task is to determine which step description corresponds "
                "to what is currently happening in the video."
            ),
            "2b": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video showing "
                "the action currently being performed.\n"
                "2. INSTRUCTION DESCRIPTIONS: Text descriptions of candidate steps for the "
                "NEXT assembly step.\n"
                "Your task is to identify which description shows the step that comes "
                "AFTER the action shown in the video. You need to understand what is currently "
                "being done and reason about what logically follows."
            ),
            "2c": (
                "You are evaluating a furniture assembly task. You will see three instruction "
                "step descriptions displayed in a shuffled (random) order.\n"
                "Your task is to determine the correct chronological assembly order of these "
                "three steps by analyzing the assembly progression described in each."
            ),
        }
    else:
        contexts = {
            "1a": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames captured from a real-world assembly video, "
                "showing a person physically assembling furniture.\n"
                "2. INSTRUCTION DIAGRAMS: Wordless assembly instruction diagrams (like IKEA manuals) "
                "that illustrate individual assembly steps using schematic drawings.\n"
                f"{tg_note}\n"
                "Your task is to determine which instruction diagram corresponds to the action "
                "shown in the video frames. Compare the physical action in the video against "
                "each diagram carefully."
            ),
            "1b": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video.\n"
                "2. INSTRUCTION DIAGRAM: A single wordless assembly instruction diagram.\n"
                f"{tg_note}\n"
                "Your task is to judge whether the action shown in the video frames matches "
                "the assembly step depicted in the instruction diagram."
            ),
            "1c": (
                "You are evaluating a furniture assembly task. You will see two sets of video frames "
                "(Clip A and Clip B) from different recordings of the same furniture product being assembled. "
                "Your task is to determine whether both clips show the same assembly step or different steps."
            ),
            "2a": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video showing "
                "the current state of assembly.\n"
                "2. ALL INSTRUCTION DIAGRAMS: The complete set of wordless instruction diagrams "
                "for this product, shown in order (Step 1, Step 2, etc.).\n"
                f"{tg_note}\n"
                "Your task is to determine which step in the instruction sequence corresponds "
                "to what is currently happening in the video."
            ),
            "2b": (
                "You are evaluating a furniture assembly task. You will see:\n"
                "1. VIDEO FRAMES: Sequential frames from a real-world assembly video showing "
                "the action currently being performed.\n"
                "2. INSTRUCTION DIAGRAMS: Candidate diagrams for the NEXT assembly step.\n"
                f"{tg_note}\n"
                "Your task is to identify which instruction diagram shows the step that comes "
                "AFTER the action shown in the video. You need to understand what is currently "
                "being done and reason about what logically follows."
            ),
            "2c": (
                "You are evaluating a furniture assembly task. You will see three instruction "
                "diagrams (wordless assembly manual images) displayed in a shuffled (random) order.\n"
                f"{tg_note}\n"
                "Your task is to determine the correct chronological assembly order of these "
                "three diagrams by analyzing the assembly progression shown in each."
            ),
        }
    return contexts.get(qtype, "")


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------
def build_prompt_and_images(q, setting, data_dir=None):
    """Build interleaved content list for a given question and setting.

    Parameters
    ----------
    q : dict
        A single question entry from the QA JSON.
    setting : str
        One of ``"baseline"``, ``"text_grounding"``, ``"text_only"``.
    data_dir : str or Path, optional
        Root data directory. Defaults to ``get_data_dir()``.

    Returns
    -------
    content : list[dict]
        List of ``{"type": "text", "text": ...}`` or
        ``{"type": "image", "image": PIL.Image}`` dicts.
    all_images : list[PIL.Image]
        Flat list of PIL images (for processor calls that need it).
    """
    _ensure_imports()
    from .utils import Image  # use the lazily imported PIL.Image

    if data_dir is None:
        data_dir = get_data_dir()
    data_dir = Path(data_dir)

    qtype = q["type"]
    content = []  # interleaved content list
    all_images = []

    def add_text(text):
        content.append({"type": "text", "text": text})

    def add_image(path):
        p = Path(path)
        # Resolve relative paths against data_dir
        if not p.is_absolute():
            p = data_dir / p
        img = Image.open(p).convert("RGB")
        content.append({"type": "image", "image": img})
        all_images.append(img)

    # --- System context ---
    system_ctx = _build_system_context(qtype, setting)
    if system_ctx:
        add_text(system_ctx)

    # --- Video frames ---
    video_frames_keys = []
    if qtype == "1c":
        video_frames_keys = [("video_frames_a", "Video clip A"), ("video_frames_b", "Video clip B")]
    elif qtype == "2c":
        video_frames_keys = []  # No video for 2c
    else:
        video_frames_keys = [("video_frames", "Video frames")]

    for vk, label in video_frames_keys:
        frame_paths = q.get(vk, [])
        if frame_paths:
            add_text(f"\n{label} ({len(frame_paths)} frames):")
            for fp in frame_paths:
                add_image(fp)

    # --- Manual images / text descriptions ---
    if setting == "text_only":
        # Replace all diagram images with text descriptions only
        if qtype == "1b":
            product = q["product"]
            sid = q.get("manual_step_id", q["metadata"].get("shown_step"))
            if sid is not None:
                tg = get_text_grounding(product, sid)
                if tg:
                    add_text(f"\nInstruction step description:\n{tg}")
                else:
                    add_text(f"\nInstruction step: Step {sid + 1}")

        if qtype == "2a" and "manual_images" in q:
            manual_imgs = q["manual_images"]
            add_text("\nComplete instruction sequence for this product:")
            for sid in sorted(manual_imgs.keys(), key=int):
                product = q["product"]
                tg = get_text_grounding(product, int(sid))
                if tg:
                    add_text(f"Step {int(sid) + 1}: {tg}")
                else:
                    add_text(f"Step {int(sid) + 1}: (no description available)")

        if qtype == "2c" and "step_images" in q:
            add_text("\nShuffled instruction step descriptions:")
            for si in q["step_images"]:
                product = q["product"]
                sid = si.get("step_id")
                if sid is not None:
                    tg = get_text_grounding(product, sid)
                    if tg:
                        add_text(f"{si['label']}: {tg}")
                    else:
                        add_text(f"{si['label']}: Step {sid + 1} (no description available)")
                else:
                    add_text(f"{si['label']}: (unknown step)")
    else:
        if qtype == "1b" and "manual_step_image" in q:
            add_text("\nInstruction diagram:")
            add_image(q["manual_step_image"])
            if setting == "text_grounding":
                product = q["product"]
                sid = q.get("manual_step_id", q["metadata"].get("shown_step"))
                if sid is not None:
                    tg = get_text_grounding(product, sid)
                    if tg:
                        add_text(f"(Description of the diagram above: {tg})")

        if qtype == "2a" and "manual_images" in q:
            manual_imgs = q["manual_images"]
            add_text("\nComplete instruction sequence for this product:")
            for sid in sorted(manual_imgs.keys(), key=int):
                add_text(f"Step {int(sid) + 1}:")
                add_image(manual_imgs[sid])
                if setting == "text_grounding":
                    product = q["product"]
                    tg = get_text_grounding(product, int(sid))
                    if tg:
                        add_text(f"(Description of the diagram above: {tg})")

        if qtype == "2c" and "step_images" in q:
            add_text("\nShuffled instruction diagrams:")
            for si in q["step_images"]:
                add_text(f"{si['label']}:")
                add_image(si["image"])
                if setting == "text_grounding":
                    product = q["product"]
                    sid = si.get("step_id")
                    if sid is not None:
                        tg = get_text_grounding(product, sid)
                        if tg:
                            add_text(f"(Description of the diagram above: {tg})")

    # --- Question ---
    add_text(f"\nQuestion: {q['question']}\n")

    # --- Options (interleaved for image options) ---
    for opt in q["options"]:
        if "text" in opt:
            add_text(f"{opt['label']}) {opt['text']}")
        elif "image" in opt:
            if setting == "text_only":
                product = q["product"]
                sid = opt.get("step_id")
                if sid is not None:
                    tg = get_text_grounding(product, sid)
                    if tg:
                        add_text(f"{opt['label']}) {tg}")
                    else:
                        add_text(f"{opt['label']}) Step {sid + 1}")
                else:
                    add_text(f"{opt['label']}) (no description)")
            else:
                # Image option: text label immediately followed by the image
                add_text(f"{opt['label']})")
                add_image(opt["image"])
                # Add text grounding for this option if applicable
                if setting in ("text_grounding",) and qtype in ("1a", "2b"):
                    product = q["product"]
                    sid = opt.get("step_id")
                    if sid is not None:
                        tg = get_text_grounding(product, sid)
                        if tg:
                            add_text(f"(Description of the diagram above: {tg})")
        else:
            add_text(f"{opt['label']}) {opt.get('step_id', '?')}")

    add_text("\nAnswer with the letter first, then briefly explain. Format: 'Answer: X. Reason: ...'")

    return content, all_images
