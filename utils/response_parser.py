import re
import string

MCQ_LETTERS = list(string.ascii_uppercase)

# Patterns the model might output even after strict instructions
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-T])\b")  # A–T covers up to 20 candidates


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def parse_zero_shot(raw: str) -> str:
    raw = _strip_think_tags(raw)
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def parse_mcq(raw: str, candidates: list[str]) -> tuple[str, str | None]:
    raw = _strip_think_tags(raw)

    # Look for the first standalone letter A–T in the output
    match = _LETTER_RE.search(raw)
    if match:
        letter = match.group(1)
        idx = MCQ_LETTERS.index(letter)
        if idx < len(candidates):
            return candidates[idx], letter

    # Fallback: if the model output the full title 
    # Check if the raw output (stripped) matches any candidate closely
    raw_stripped = raw.strip().lower()
    for candidate in candidates:
        if candidate.lower() == raw_stripped:
            return candidate, None

    return "", None


def parse_response(
    raw: str,
    mode: str,
    candidates: list[str] | None = None,
) -> dict:
    if mode == "zero_shot":
        predicted = parse_zero_shot(raw)
        return {"predicted": predicted, "letter": None, "raw_output": raw}

    elif mode == "mcq":
        if not candidates:
            raise ValueError("candidates must be provided in mcq mode")
        predicted, letter = parse_mcq(raw, candidates)
        return {"predicted": predicted, "letter": letter, "raw_output": raw}

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'zero_shot' or 'mcq'.")