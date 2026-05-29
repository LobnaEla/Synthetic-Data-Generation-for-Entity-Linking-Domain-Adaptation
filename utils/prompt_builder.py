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
    descriptions: list[str] | None = None,
) -> str:
    if not candidates:
        return build_prompt_zero_shot(context, mention, entity_type)

    # Build candidate lines, optionally with descriptions (strongly recommended by paper)
    def candidate_line(i: int, title: str) -> str:
        letter = MCQ_LETTERS[i]
        if descriptions and i < len(descriptions) and descriptions[i]:
            return f"{letter}. {title} — {descriptions[i]}"
        return f"{letter}. {title}"

    options_str = "\n".join(
        candidate_line(i, title)
        for i, title in enumerate(candidates[:len(MCQ_LETTERS)])
    )

    letters = MCQ_LETTERS[:len(candidates[:len(MCQ_LETTERS)])]
    valid_letters = ", ".join(letters)

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
3. Consider each candidate carefully — resolve abbreviations, acronyms, aliases, nicknames, and ambiguous surface forms using the surrounding context.
4. Eliminate candidates that contradict information in the TEXT.
5. Select the single best matching candidate.

Output your answer as a single letter ({valid_letters}).
Format: "Answer: <letter>"
"""


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    candidates = []
    for line in prompt.splitlines():
        line = line.strip()
        if len(line) >= 3 and line[0].isupper() and line[1] == "." and line[2] == " ":
            candidates.append(line[3:])
    return candidates