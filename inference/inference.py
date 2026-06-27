"""A/B inference: base Qwen2.5-1.5B vs base+LoRA."""
import torch, time, json, os, sys
# Disable torch.compile overhead
import torch._dynamo
torch._dynamo.config.disable = True

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROMPTS = [
    {"type": "事实问答", "prompt": "Instruction: 什么是Kubernetes Operator?\nResponse:"},
    {"type": "代码生成", "prompt": "Instruction: 写一段Python代码，使用PyTorch定义一个简单的神经网络\nResponse:"},
    {"type": "推理", "prompt": "Instruction: 我有8张GPU，每张16GB显存，想训练一个7B参数的模型。请问batch size应该怎么估算？\nResponse:"},
    {"type": "翻译", "prompt": "Instruction: 把下面这句话翻译成英文：分布式训练的核心是通信和计算的重叠\nResponse:"},
    {"type": "常识", "prompt": "Instruction: 解释一下为什么天空是蓝色的\nResponse:"},
    {"type": "对话", "prompt": "Instruction: 你好，请介绍一下你自己\nResponse:"},
]

model_name = "Qwen/Qwen2.5-1.5B"
lora_path = "/lora-output"
device = "cuda"

sys.stderr.write(f"Loading base model from {model_name}...\n")
sys.stderr.flush()

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                            device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
sys.stderr.write("Base model loaded.\n")
sys.stderr.flush()

sys.stderr.write("Loading LoRA adapter...\n")
sys.stderr.flush()
lora = PeftModel.from_pretrained(base, lora_path)
sys.stderr.write("LoRA loaded.\n")
sys.stderr.flush()

def generate(model, prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - t0
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip(), elapsed

results = []
for i, item in enumerate(PROMPTS):
    sys.stderr.write(f"\n[{i+1}/{len(PROMPTS)}] {item['type']}\n")
    sys.stderr.flush()

    base_resp, base_time = generate(base, item["prompt"])
    lora_resp, lora_time = generate(lora, item["prompt"])

    print(f"\n{'='*60}")
    print(f"[{item['type']}] {item['prompt'][:70]}")
    print(f"\n--- Base [{base_time:.1f}s] ---")
    print(base_resp[:400])
    print(f"\n--- Base+LoRA [{lora_time:.1f}s] ---")
    print(lora_resp[:400])
    sys.stdout.flush()

    results.append({
        "type": item["type"], "prompt": item["prompt"],
        "base": base_resp, "lora": lora_resp,
        "base_time": base_time, "lora_time": lora_time,
    })

with open("/output/inference_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

sys.stderr.write("Done.\n")