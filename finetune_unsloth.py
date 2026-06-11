"""Fine-tune Qwen3 (or compatible) with Unsloth + QLoRA on entity-linking data.

Input: JSONL where each line has `prompt` (str) and `ground_truth_id` (str|int).
Each example becomes a (user -> assistant) supervised pair, formatted with the
model's chat template. Only the assistant response is used for loss computation
(`train_on_responses_only`).

Usage:
    python finetune_unsloth.py path/to/train.jsonl path/to/output_dir
"""

# Unsloth must be imported before trl/transformers/peft: it patches them at import time.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse

import torch
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune an LLM on entity-linking data with Unsloth + QLoRA."
    )
    p.add_argument("train_data", help="Path to JSONL with fields `prompt` and `ground_truth_id`.")
    p.add_argument("output_dir", help="Directory for checkpoints and the final LoRA adapter.")
    p.add_argument("--val_data", default=None,
                   help="Path to a versioned val JSONL (made by make_splits.py). "
                        "If omitted, a random 1%% split is carved from train_data.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Base model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--base_adapter", default=None,
                   help="Path to an existing LoRA adapter to continue training from.")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the last checkpoint in output_dir.")
    p.add_argument("--max_length", type=int, default=5120)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Cap total training steps (smoke tests). -1 = full epochs.")
    p.add_argument("--report_to", choices=["wandb", "none"], default="wandb")
    p.add_argument("--wandb_project", default="ed-finetuning")
    p.add_argument("--wandb_run", default="ed-unsloth-qlora")
    return p.parse_args()


def build_dataset(path, tokenizer, seed, val_path=None):
    def to_chat(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": str(example["ground_truth_id"])},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(to_chat, remove_columns=ds.column_names)

    if val_path:
        val = load_dataset("json", data_files=val_path, split="train")
        val = val.map(to_chat, remove_columns=val.column_names)
        return ds, val

    split = ds.train_test_split(test_size=0.01, seed=seed)
    return split["train"], split["test"]


def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        dtype=None,
    )

    if args.base_adapter:
        print(f"Loading existing LoRA adapter from {args.base_adapter}")
        model.load_adapter(args.base_adapter, is_trainable=True)
    else:
        print("Initializing a fresh LoRA adapter")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    model.print_trainable_parameters()

    train_ds, eval_ds = build_dataset(
        args.train_data, tokenizer, args.seed, val_path=args.val_data
    )

    if args.report_to == "wandb":
        wandb.init(project=args.wandb_project, name=args.wandb_run)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_length=args.max_length,
        dataset_text_field="text",
        dataset_num_proc=1,  # Windows: multiprocessing dataset workers hang
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=True,
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Qwen3 chat template: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f"{args.output_dir}/final_adapter")


if __name__ == "__main__":
    main()
