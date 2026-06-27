"""Qwen2.5-1.5B LoRA fine-tuning with QLoRA (4-bit) for V100 16GB."""
import os, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data", default="tatsu-lab/alpaca")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()

    # 4-bit quantization for V100 16GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {args.data}")
    dataset = load_dataset(args.data, split="train")
    if len(dataset) > 5000:
        dataset = dataset.select(range(5000))

    def tokenize(examples):
        texts = []
        for inst, inp, out in zip(examples.get("instruction",[""]), examples.get("input",[""]), examples.get("output",[""])):
            text = f"Instruction: {inst}\nInput: {inp}\nResponse: {out}"
            texts.append(text)
        return tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    print("Training started...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()