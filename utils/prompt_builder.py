import string

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
    context: str,
    mention: str,
    entity_type: str,
    candidates: list[str],
) -> str:
    if not candidates:
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


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    candidates = []
    for line in prompt.splitlines():
        line = line.strip()
        if len(line) >= 3 and line[0].isupper() and line[1] == "." and line[2] == " ":
            candidates.append(line[3:])
    return candidates