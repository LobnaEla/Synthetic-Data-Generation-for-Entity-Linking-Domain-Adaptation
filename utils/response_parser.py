import re

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_NUM_RE = re.compile(r"(?:answer)\s*[:=]\s*\"?(\d+)\"?", re.IGNORECASE)
_STARTSWITH_NUM_RE = re.compile(r"^\s*\"?(\d+)\"?\s*[.\s,)]", re.MULTILINE)
_STANDALONE_NUM_RE = re.compile(r"\b(\d+)\b")


def parse_mcq_response(raw: str, candidates: list[str]) -> dict:
    if not candidates:
        raise ValueError("candidates required")

    cleaned = _THINK_TAG_RE.sub("", raw).strip()
    cleaned = cleaned.replace('"', '').replace("'", "")

    for regex in (_ANSWER_NUM_RE, _STARTSWITH_NUM_RE, _STANDALONE_NUM_RE):
        match = regex.search(cleaned)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(candidates):
                return {"predicted": candidates[idx], "index": idx, "raw_output": raw}

    for candidate in candidates:
        if candidate.lower() == cleaned.lower():
            return {"predicted": candidate, "index": None, "raw_output": raw}

    return {"predicted": "", "index": None, "raw_output": raw}