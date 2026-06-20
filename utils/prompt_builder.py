import string

<<<<<<< HEAD
# Letters used to label MCQ candidates: A, B, C, ..., T (up to 20 candidates)
MCQ_LETTERS = list(string.ascii_uppercase)


def build_prompt_zero_shot(context: str, mention: str, entity_type: str) -> str:
    return f"""/no_think
You are a high-precision English Wikipedia Entity Linking specialist.

Your task is to disambiguate one entity mention in context.

You will receive:
TEXT: a passage containing the mention
MENTION: the exact surface form to link
ENTITY TYPE: the semantic type of the mention

Return the single English Wikipedia article title that best corresponds to the MENTION in the given TEXT.

Use the TEXT context to resolve aliases, abbreviations, acronyms, nicknames, surnames, demonyms, adjectival forms, shortened forms, informal references, and ambiguous mentions.

Rules:
- Output exactly one English Wikipedia article title.
- Output only the title.
- Do not explain.
- Do not output JSON.
- Do not output Markdown.
- Do not output a sentence.
- Do not output a URL.
- Do not use underscores.
- Use spaces between words.
- Preserve the official Wikipedia capitalization.
- If the mention is already the correct article title, return it unchanged.
- If the mention is shortened or ambiguous, resolve it to the most specific entity supported by the TEXT.
- If parenthetical disambiguation is required, include it.
- Do not invent a non-existing title.
- Do not output placeholders.
- Never output braces or template variables.
- If uncertain, output the single most plausible existing English Wikipedia article title supported by the TEXT.

TEXT:
{context}

MENTION:
{mention}

ENTITY TYPE:
{entity_type}

TITLE:"""


def build_prompt_mcq(
=======
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


def build_mcq_prompt(
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
    context: str,
    mention: str,
    entity_type: str,
    candidates: list[str],
) -> str:
    if not candidates:
<<<<<<< HEAD
        # Fallback to zero-shot if retrieval returned nothing
        return build_prompt_zero_shot(context, mention, entity_type)

    options_str = "\n".join(
        f"{MCQ_LETTERS[i]}. {title}" for i, title in enumerate(candidates[:len(MCQ_LETTERS)])
    )

    return f"""/no_think
You are a precise English Wikipedia Entity Linking system.

Your task is to identify which candidate Wikipedia article best corresponds to the entity mention in the given text.

Input:
TEXT: {context}
MENTION: {mention}
ENTITY_TYPE: {entity_type}

Candidates:
{options_str}

Instructions:
- Read TEXT carefully to understand what real-world entity MENTION refers to.
- Use ENTITY_TYPE to avoid selecting the wrong kind of entity.
- Resolve abbreviations, acronyms, aliases, nicknames, and ambiguous mentions.
- Select the single best matching candidate.

Output constraints:
- Output ONLY the letter of your choice (A, B, C, ...).
- Do not explain.
- Do not output the full title.
- Do not add punctuation.

Answer:"""
=======
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
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    candidates = []
    for line in prompt.splitlines():
        line = line.strip()
<<<<<<< HEAD
        if len(line) >= 3 and line[0].isupper() and line[1] == "." and line[2] == " ":
            candidates.append(line[3:])
=======
        # Match "0. Title" or "0. Title - Description"
        if len(line) >= 3 and line[0].isdigit() and ". " in line:
            # Extract everything after "N. ", strip description if present
            after_index = line.split(". ", 1)[1]
            # Remove description part (" - ...") to get just the title
            title = after_index.split(" - ", 1)[0]
            candidates.append(title)
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
    return candidates