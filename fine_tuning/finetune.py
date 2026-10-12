import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig
import wandb

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def format_to_messages(example, disable_thinking=False):
    return {
        "prompt": example["prompt"] + (" /no_think" if disable_thinking else ""),
        "completion": ("<think>\n\n</think>\n\n" if disable_thinking else "")
        + str(example["ground_truth_id"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on the entity disambiguation candidate selection task."
    )
    parser.add_argument(
        "train_data",
        type=str,
        help="Path to the training data JSONL file.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the fine-tuned model and checkpoints.",
    )
    parser.add_argument(
        "--base_adapter",
        type=str,
        default=None,
        help="Path to a pre-trained LoRA adapter to start fine-tuning from.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume training from the last checkpoint in the output directory.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=5120,
        help="Maximum sequence length for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Base model name or path to fine-tune (default: Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0e-4,
        help="Learning rate for training (default: 1.0e-4).",
    )

    args = parser.parse_args()
    train_data_path = args.train_data
    output_dir = args.output_dir
    base_adapter = args.base_adapter
    resume_training = args.resume
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    max_length = args.max_length
    epochs = args.epochs
    model_name = args.model
    disable_thinking = args.disable_thinking
    learning_rate = args.learning_rate

    dataset = load_dataset("json", data_files=train_data_path, split="train")
    dataset = dataset.map(
        lambda x: format_to_messages(x, disable_thinking=disable_thinking),
        remove_columns=dataset.column_names,
    )
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if base_adapter:
        print(
            f"Loading existing LoRA adapter from {base_adapter} to continue training..."
        )
        model = PeftModel.from_pretrained(model, base_adapter, is_trainable=True)
    else:
        print("Initializing a fresh LoRA adapter...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=128,
            lora_alpha=256,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    wandb.init(project="ed-finetuning", name="ed-finetune-run")

    training_args = SFTConfig(
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        output_dir=output_dir,
        max_length=max_length,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=True,
        report_to="wandb",
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_training)
