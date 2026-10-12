import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    """Load a full HF model, or a PEFT/LoRA adapter directory (detected via
    adapter_config.json: the base model is loaded first, then the adapter is
    applied on top)."""
    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftModel

        with open(adapter_config_path, encoding="utf-8") as f:
            base_model_name = json.load(f)["base_model_name_or_path"]

        print(f"[LLM] Loading base model {base_model_name} for adapter {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print(f"[LLM] Applying adapter {model_name} ...")
        model = PeftModel.from_pretrained(base_model, model_name).eval()
    else:
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
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 2048,
) -> list[str]:
    """Generate one response per prompt, in batch."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


def _is_oom_error(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def generate_batch_safe(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 2048,
) -> list[tuple[str, str]]:
    """Like generate_batch, but recovers from CUDA OOM by halving the batch instead
    of losing every prompt in it. A single prompt that still OOMs on its own is
    reported as an error for that example only. Returns (raw_output, error) pairs
    aligned 1:1 with prompts.
    """
    try:
        outputs = generate_batch(prompts, tokenizer, model, max_new_tokens)
        return [(o, "") for o in outputs]
    except RuntimeError as e:
        if not _is_oom_error(e):
            raise
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            return [("", f"CUDA OOM: {e}")]
        mid = len(prompts) // 2
        return (
            generate_batch_safe(prompts[:mid], tokenizer, model, max_new_tokens)
            + generate_batch_safe(prompts[mid:], tokenizer, model, max_new_tokens)
        )