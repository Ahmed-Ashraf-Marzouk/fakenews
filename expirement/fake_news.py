import pandas as pd
import re
import torch
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('/mnt/bigdisk/python_libs')

# tqdm (safe fallback)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# Optional deps
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import InferenceClient
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    InferenceClient = None

# =====================================================================
# Utils
# =====================================================================
def prefer_bf16():
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

def get_dtype():
    return torch.bfloat16 if prefer_bf16() else torch.float16

_LOCAL_MODEL_CACHE = {}  # {model_id: (tokenizer, model)}

# =====================================================================
# Few-shot generator
# =====================================================================
def generate_shots(no_shots):
    # Match the main() datasets path so there’s no mismatch
    df = pd.read_csv(f'../datasets/{args.task}/train.csv')
    df['fake_flag'] = df['fake_flag'].astype(str)
    num_per_class = max(no_shots // 2, 0)

    real_samples = df[df['fake_flag'] == '0'].iloc[:num_per_class]
    fake_samples = df[df['fake_flag'] == '1'].iloc[:num_per_class]

    balanced_samples = pd.concat([real_samples, fake_samples])
    if len(balanced_samples) > 0:
        balanced_samples = balanced_samples.sample(frac=1, random_state=42)

    shots = '\n\n'.join(
        f"Text: {row['claim_s']}\nAnswer: {row['fake_flag']}"
        for _, row in balanced_samples.iterrows()
    )
    return shots

# =====================================================================
# Prompt builder (plain text only)
# =====================================================================
def build_prompt(sentence, template=None, no_shots=0, task=None):
    if template:
        if no_shots:
            return template.format(sentence=sentence, shots=generate_shots(no_shots))
        return template.format(sentence=sentence)
    prompt = f"""You are a fake news classifier. Given a news headline, output only the classification result:

- Output **0** if the news is **real**
- Output **1** if the news is **fake**
- Output must be the **one digit only (0 or 1)**, no explanation or symbols.

Now classify the following:

Text: {sentence}
Answer:"""
    return prompt

# =====================================================================
# HF local: load once
# =====================================================================
def load_local_model_once(model_id):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required for provider 'hf-local'")

    if model_id in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_id]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=get_dtype(),
        trust_remote_code=True,
    )
    _LOCAL_MODEL_CACHE[model_id] = (tok, mdl)
    return tok, mdl

def hf_local_generate(prompt, model_id, tokenizer=None, model=None, max_new_tokens=10):
    if tokenizer is None or model is None:
        tokenizer, model = load_local_model_once(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Plain decoding; strip the prompt if present
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.replace(prompt, "").strip()

# =====================================================================
# HF API path (unchanged)
# =====================================================================
def hf_api_generate(prompt, model_id):
    if InferenceClient is None:
        raise ImportError("huggingface_hub is required for provider 'hf-api'")
    client = InferenceClient(model_id)
    return client.text_generation(prompt, max_new_tokens=1, temperature=0.0).strip()

# =====================================================================
# Unified completion handler
# =====================================================================
def get_completion(params, prompt, tokenizer=None, model_obj=None):
    provider = params["provider"]
    model = params["model"]

    if provider == "ollama":
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "seed": 42},
            stream=False
        )
        return response["message"]["content"].strip()

    elif provider == "openai":
        import openai
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1
        )
        return response['choices'][0]['message']['content'].strip()

    elif provider == "hf-local":
        # NO chat template: always plain prompt path
        return hf_local_generate(prompt, model_id=model, tokenizer=tokenizer, model=model_obj, max_new_tokens=2)

    elif provider == "hf-api":
        return hf_api_generate(prompt, model)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

# =====================================================================
# Metrics + inference
# =====================================================================
def calculate_metrics(df, params, template=None, no_shots=0, tokenizer=None, model=None, task=None):
    true_labels, predicted_labels, preds_all, error_indices = [], [], [], []
    raw_outputs, parse_status = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Running inference'):
        sentence = re.sub(r'[\W_]+', ' ', row['claim_s'])
        actual = row['fake_flag']
        prompt = build_prompt(sentence, template=template, no_shots=no_shots, task=task)
        print(f"\n[Prompt idx {idx}]:\n{prompt}\n")

        try:
            raw = get_completion(params, prompt, tokenizer=tokenizer, model_obj=model)
            raw_outputs.append(raw)
        except Exception as e:
            print(f"[ERROR] idx {idx} – failed to get completion: {e}")
            error_indices.append(idx)
            preds_all.append(None)
            raw_outputs.append(raw)
            parse_status.append("error")
            continue

        lower = (raw or "").strip().lower()
        if lower.startswith('fake'):
            val = 1
            parse_status.append("ok:text=fake")
        elif lower.startswith('real') or lower.startswith('false'):
            val = 0
            parse_status.append("ok:text=real/false")
        else:
            m = re.search(r'[01]', raw or "")
            if not m:
                print(f"[INVALID] idx {idx} – no '0' or '1' found; raw output: >>>{raw}<<<")
                error_indices.append(idx)
                preds_all.append(None)
                parse_status.append("invalid:no_digit")
                continue
            val = int(m.group(0))
            parse_status.append("ok:digit")

        true_labels.append(int(actual))
        predicted_labels.append(val)
        preds_all.append(val)

    if len(predicted_labels) == 0:
        acc = prec = rec = f1 = 0.0
    else:
        acc = accuracy_score(true_labels, predicted_labels)
        prec = precision_score(true_labels, predicted_labels, zero_division=0)
        rec = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return acc, prec, rec, f1, preds_all, error_indices, raw_outputs, parse_status

# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fake news classification using LLMs.")
    parser.add_argument("--provider", type=str, required=True, choices=["ollama", "openai", "hf-local", "hf-api"], help="Model provider")
    parser.add_argument("--model", type=str, required=True, help="Model ID or name")
    parser.add_argument("--input", type=str, default="test.csv", help="CSV file with 'claim_s' and 'fake_flag' columns")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file with predictions")
    parser.add_argument("--prompt", type=str, default=None, help="Path to a text file containing the prompt template. Use {sentence} and optionally {shots}.")
    parser.add_argument("--no_shots", type=str, default="0", choices=["0", "2", "4", "8", "16", "32"], help="Number of examples to be used")
    parser.add_argument("--task", type=str, default="ANS", choices=["ANS", "ArAiEval"])
    args = parser.parse_args()
    args.no_shots = int(args.no_shots)

    # Use the same datasets root as generate_shots()
    df = pd.read_csv(f"../datasets/{args.task}/{args.input}")

    prompt_template = None
    if args.prompt:
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    params = {"provider": args.provider, "model": args.model}

    # Preload HF-local model ONCE (no chat template anywhere)
    tok = mdl = None
    if args.provider == "hf-local":
        print(f"🔹 Loading local model once: {args.model}")
        tok, mdl = load_local_model_once(args.model)

    acc, prec, rec, f1, preds, errors, raws, status = calculate_metrics(
        df,
        params,
        template=prompt_template,
        no_shots=args.no_shots,
        task=args.task,
        tokenizer=tok,
        model=mdl
    )

    print("\n--- Classification Report ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")
    if errors:
        print(f"Errors occurred at indices: {errors}")

    if args.output is None:
        model_name = args.model.replace(":", "_")
        os.makedirs(f'./predictions/{args.provider}_{model_name}/{args.task}', exist_ok=True)
        suffix = '_cot' if args.prompt == 'prompt_cot.txt' or (args.prompt and 'cot' in args.prompt.lower()) else ''
        args.output = f"./predictions/{args.provider}_{model_name}/{args.task}/pred_s{args.no_shots}{suffix}.csv"

    df["Predicted"] = preds
    df["RawOutput"] = raws
    df["ParseStatus"] = status
    df.to_csv(args.output, index=False)

    with open(args.output, "a", encoding="utf-8") as f:
        f.write("\n--- Classification Report ---\n")
        f.write(f"Accuracy , {acc:.3f}\n")
        f.write(f"Precision, {prec:.3f}\n")
        f.write(f"Recall   , {rec:.3f}\n")
        f.write(f"F1 Score , {f1:.3f}\n")
        if errors:
            f.write(f"Errors   , {errors}\n")
