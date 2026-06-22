import re

from utils.prompt_builder import get_candidate_ids

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_ID_RE = re.compile(r"(?:answer)\s*[:=]\s*\"?([QC]\d+)\"?", re.IGNORECASE)
_STARTSWITH_ID_RE = re.compile(r"^\s*\"?([QC]\d+)\"?\s*[.\s,)]", re.MULTILINE)
_STANDALONE_QID_RE = re.compile(r"\b(Q\d+)\b")


def parse_mcq_response(
    raw: str,
    candidates: list[str],
    title_to_qid: dict[str, str] | None = None,
) -> dict:
    if not candidates:
        raise ValueError("candidates required")

    ids = get_candidate_ids(candidates, title_to_qid)
    id_to_title = dict(zip(ids, candidates))

    cleaned = _THINK_TAG_RE.sub("", raw).strip()
    cleaned = cleaned.replace('"', '').replace("'", "")

    for regex in (_ANSWER_ID_RE, _STARTSWITH_ID_RE, _STANDALONE_QID_RE):
        match = regex.search(cleaned)
        if match:
            cid = match.group(1).upper()
            if cid in id_to_title:
                return {"predicted": id_to_title[cid], "id": cid, "raw_output": raw}

    for candidate in candidates:
        if candidate.lower() == cleaned.lower():
            return {"predicted": candidate, "id": None, "raw_output": raw}

    return {"predicted": "", "id": None, "raw_output": raw}
