import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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