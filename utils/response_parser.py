import re
import string

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_NUM_RE = re.compile(r"(?:answer)\s*[:=]\s*\"?(\d+)\"?", re.IGNORECASE)
_STARTSWITH_NUM_RE = re.compile(r"^\s*\"?(\d+)\"?\s*[.\s,)]", re.MULTILINE)
_STANDALONE_NUM_RE = re.compile(r"\b(\d+)\b")


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()

def parse_zero_shot(raw: str) -> str:
    raw = _strip_think_tags(raw)
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def parse_mcq(raw: str, candidates: list[str]) -> tuple[str, int | None]:
    raw = _strip_think_tags(raw)
    cleaned = raw.replace('"', '').replace("'", "").strip()

    # Priority 1: "answer: 3" pattern
    match = _ANSWER_NUM_RE.search(cleaned)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(candidates):
            return candidates[idx], idx

    # Priority 2: line starts with a number like "3" or "3."
    match = _STARTSWITH_NUM_RE.search(cleaned)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(candidates):
            return candidates[idx], idx

    # Priority 3: any standalone number (last resort)
    match = _STANDALONE_NUM_RE.search(cleaned)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(candidates):
            return candidates[idx], idx

    # Fallback: model output the full title
    raw_lower = cleaned.lower()
    for candidate in candidates:
        if candidate.lower() == raw_lower:
            return candidate, None

    return "", None


def parse_response(raw: str, mode: str, candidates: list[str] | None = None) -> dict:
    if mode == "zero_shot":
        predicted = parse_zero_shot(raw)
        return {"predicted": predicted, "index": None, "raw_output": raw}
    elif mode == "mcq":
        if not candidates:
            raise ValueError("candidates must be provided in mcq mode")
        predicted, idx = parse_mcq(raw, candidates)
        return {"predicted": predicted, "index": idx, "raw_output": raw}
    else:
        raise ValueError(f"Unknown mode: {mode!r}")