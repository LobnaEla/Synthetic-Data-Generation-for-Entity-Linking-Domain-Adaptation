import string

def _split_context_around_mention(context: str, mention: str) -> tuple[str, str]:
    """Split context around the first occurrence of the mention.

    The training/evaluation prompt from filter_zelda.py is built from character
    offsets. In this benchmark pipeline we only receive the full context and the
    mention string, so we mark the first occurrence of the mention.
    """
    if mention and mention in context:
        before, after = context.split(mention, 1)
        return before, after

    # Fallback: keep the full context and still show the mention explicitly.
    # This avoids crashing if the mention string has been normalized differently.
    return context + "\n\nMention: ", ""


def get_candidate_ids(
    candidates: list[str],
    title_to_qid: dict[str, str] | None = None,
) -> list[str]:
    """Map each candidate title to a stable id shown to/expected from the LLM:
    its Wikidata QID when known, otherwise a positional fallback ("C0", "C1", ...).
    """
    title_to_qid = title_to_qid or {}
    return [title_to_qid.get(title) or f"C{i}" for i, title in enumerate(candidates)]


def build_mcq_prompt(
    context: str,
    mention: str,
    entity_type: str,
    candidates: list[str],
    title_to_qid: dict[str, str] | None = None,
) -> str:
    if not candidates:
        raise ValueError("candidates list is empty")

    ids = get_candidate_ids(candidates, title_to_qid)
    options_str = "\n".join(f"{cid}. {title}" for cid, title in zip(ids, candidates))

    return f"""You are an expert designed to disambiguate entities in text, taking into account the overall context and a list of entity candidates.
TEXT (the mention to disambiguate is enclosed in square brackets):
{context.replace(mention, f"[{mention}]", 1)}
MENTION: {mention}
ENTITY TYPE: {entity_type}
CANDIDATES:
{options_str}
Your task:
1. Identify what real-world entity [{mention}] refers to in the TEXT above.
2. Use the ENTITY TYPE to rule out candidates of the wrong type.
3. Consider each candidate carefully : resolve abbreviations, acronyms, aliases, nicknames, and ambiguous surface forms using the surrounding context.
4. Eliminate candidates that contradict information in the TEXT.
5. Select the single best matching candidate.
Please show your choice with only the candidate id, e.g., "answer: Q42".
"""


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    candidates = []
    for line in prompt.splitlines():
        line = line.strip()
        # Match "Q42. Title", "C0. Title", or "... - Description"
        if ". " in line:
            cid, after_id = line.split(". ", 1)
            if cid and (cid.startswith("Q") or cid.startswith("C")) and cid[1:].isdigit():
                # Remove description part (" - ...") to get just the title
                title = after_id.split(" - ", 1)[0]
                candidates.append(title)
    return candidates