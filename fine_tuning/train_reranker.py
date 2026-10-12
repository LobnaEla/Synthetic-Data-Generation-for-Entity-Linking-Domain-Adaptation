import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb


class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return (loss, outputs) if return_outputs else loss


def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        # padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = examples["label"]
    return tokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reranker model.")
    parser.add_argument(
        "train_data",
        type=str,
        help="Path to the training data JSONL file.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the trained model and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume training from the last checkpoint in the output directory.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device for training and evaluation (default: 2).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients before performing an optimizer step (default: 8).",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Enable 4-bit quantization"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0e-4,
        help="Learning rate for training (default: 2e-4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    args = parser.parse_args()

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.quantize:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=128,
        lora_alpha=256,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_data)["train"]

    max_length = 4096

    wandb.init(project="qwen3-reranker-finetuning", name="qwen3-reranker-finetune-run")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"]
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=True,
        report_to="wandb",
        eval_on_start=True,
    )

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    trainer.save_model(f"{args.output_dir}/final_model")
    print("Training complete and model saved!")
