from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.response_parser import parse_response


def load_model(model_name: str):
    print(f"[LLM] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    print(f"[LLM] Loaded on {device}")
    return tokenizer, model


def generate_batch(
    prompt: str,
    tokenizer,
    model,
    n_samples: int = 3,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> list[str]:
    inputs = tokenizer(
        [prompt], return_tensors="pt", padding=True,
        truncation=True, max_length=4096
    ).to(model.device)

    outputs_list = []
    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id
            )
        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        outputs_list.append(text)

    return outputs_list


def majority_vote(raw_outputs: list[str], candidates: list[str]) -> dict:
    predictions = []
    for raw in raw_outputs:
        parsed = parse_response(raw, mode="mcq", candidates=candidates)
        if parsed["predicted"]:
            predictions.append(parsed["predicted"])

    if not predictions:
        return {"predicted": "", "index": None, "raw_outputs": raw_outputs, "n_agree": 0}

    counter = Counter(predictions)
    best, n_agree = counter.most_common(1)[0]

    return {
        "predicted": best,
        "index": candidates.index(best) if best in candidates else None,
        "raw_outputs": raw_outputs,
        "n_agree": n_agree,
        "n_samples": len(raw_outputs),
    }