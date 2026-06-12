import string


def build_mcq_prompt(
    context: str,
    mention: str,
    entity_type: str,
    candidates: list[str],
) -> str:
    if not candidates:
        raise ValueError("candidates list is empty")

    options_str = "\n".join(f"{i}. {title}" for i, title in enumerate(candidates))

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
Please show your choice with only the index, e.g., "answer: 3".
"""


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    candidates = []
    for line in prompt.splitlines():
        line = line.strip()
        # Match "0. Title" or "0. Title - Description"
        if len(line) >= 3 and line[0].isdigit() and ". " in line:
            # Extract everything after "N. ", strip description if present
            after_index = line.split(". ", 1)[1]
            # Remove description part (" - ...") to get just the title
            title = after_index.split(" - ", 1)[0]
            candidates.append(title)
    return candidates