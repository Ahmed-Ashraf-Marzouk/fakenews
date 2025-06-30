import pandas as pd
import re
import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import InferenceClient
except ImportError:
    pass

# Prompt builder
def build_prompt(sentence, template=None):
    if template:
        return template.format(sentence=sentence)
    return f"""You are a fake news classifier. Given a news headline, output only the classification result:

- Output **0** if the news is **real**
- Output **1** if the news is **fake**
- Output must be the **one digit only (0 or 1)**, no explanation or symbols.

Now classify the following:

Text: {sentence}
Answer:"""

# HF local
def hf_local_generate(prompt, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# HF API
def hf_api_generate(prompt, model_id):
    client = InferenceClient(model_id)
    return client.text_generation(prompt, max_new_tokens=1, temperature=0.0).strip()

# Completion handler
def get_completion(params, prompt):
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
        return hf_local_generate(prompt, model)

    elif provider == "hf-api":
        return hf_api_generate(prompt, model)

    else:
        raise ValueError("Unsupported provider")

# Metrics
def calculate_metrics(df, params, template=None):
    true_labels, predicted_labels = [], []

    for index, row in df.iterrows():
        sentence = re.sub(r'[\W_]+', ' ', row['claim_s'])
        actual = row['fake_flag']
        prompt = build_prompt(sentence, template=template)

        try:
            output = get_completion(params, prompt)
            cleaned = re.sub(r"[^01]", "", output)
            print(f"[{index}] Output: '{output}' → '{cleaned}' | True: {actual}")

            if cleaned in {"0", "1"}:
                predicted = int(cleaned)
                true_labels.append(actual)
                predicted_labels.append(predicted)
            else:
                print(f"⚠️ Skipping index {index}: invalid output '{output}'")
        except Exception as e:
            print(f"❌ Error at index {index}: {e}")
            continue

    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return acc, prec, rec, f1, predicted_labels

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fake news classification using LLMs.")
    parser.add_argument("--provider", type=str, required=True, choices=["ollama", "openai", "hf-local", "hf-api"], help="Model provider")
    parser.add_argument("--model", type=str, required=True, help="Model ID or name")
    parser.add_argument("--input", type=str, default="test.csv", help="CSV file with 'claim_s' and 'fake_flag' columns")
    parser.add_argument("--output", type=str, default="predictions_with_labels.csv", help="Output CSV file with predictions")
    parser.add_argument("--prompt", type=str, default=None, help="Path to a text file containing the prompt template. Use {sentence} as placeholder.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    prompt_template = None
    if args.prompt:
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    params = {
        "provider": args.provider,
        "model": args.model
    }

    acc, prec, rec, f1, preds = calculate_metrics(df, params, template=prompt_template)

    print("\n--- Classification Report ---")
    print(f"Accuracy : {round(acc, 3)}")
    print(f"Precision: {round(prec, 3)}")
    print(f"Recall   : {round(rec, 3)}")
    print(f"F1 Score : {round(f1, 3)}")

    df["Predicted"] = preds
    df.to_csv(args.output, index=False)
