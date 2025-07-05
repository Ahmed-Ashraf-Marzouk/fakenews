import pandas as pd
import re
import torch
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import InferenceClient
except ImportError:
    pass

# Shots generator
def generate_shots(no_shots):
    # Load the generated prompts DataFrame
    # Load CSV
    df = pd.read_csv(f'./datasets/{args.task}/train.csv')

    # Convert label column to string
    df['fake_flag'] = df['fake_flag'].astype(str)

    # Number of examples per class
    num_per_class = no_shots // 2

    # Get balanced samples
    real_samples = df[df['fake_flag'] == '0'].iloc[:num_per_class]
    fake_samples = df[df['fake_flag'] == '1'].iloc[:num_per_class]

    # Combine and shuffle
    balanced_samples = pd.concat([real_samples, fake_samples])
    balanced_samples = balanced_samples.sample(frac=1, random_state=42)

    # Convert each row to the desired format
    shots = '\n\n'.join(
        f"Text: {row['claim_s']}\nAnswer: {row['fake_flag']}"
        for _, row in balanced_samples.iterrows()
    )
    return shots
# Prompt builder
def build_prompt(sentence, template=None, no_shots=0):
    if template:
        if no_shots:
            return template.format(sentence=sentence, shots=generate_shots(no_shots)) # few-shots
        return template.format(sentence=sentence) # COT or any other templates
    # zero-shot
    prompt = f"""You are a fake news classifier. Given a news headline, output only the classification result:

- Output **0** if the news is **real**
- Output **1** if the news is **fake**
- Output must be the **one digit only (0 or 1)**, no explanation or symbols.

Now classify the following:

Text: {sentence}
Answer:"""
    
    return prompt

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
def calculate_metrics(df, params, template=None, no_shots=0):
    true_labels, predicted_labels = [], []

    for index, row in df.iterrows():
        sentence = re.sub(r'[\W_]+', ' ', row['claim_s'])
        actual = row['fake_flag']
        prompt = build_prompt(sentence, template=template, no_shots=no_shots)
        print(prompt)
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
    parser.add_argument("--output", type=str, default=None, help="Output CSV file with predictions")
    parser.add_argument("--prompt", type=str, default=None, help="Path to a text file containing the prompt template. Use {sentence} as placeholder.")
    parser.add_argument("--no_shots", type=str, default="0", choices=["0", "2", "4", "8", "16", "32"], help="Number of examples to be used" )
    parser.add_argument("--task", type=str, default="ANS", choices=["ANS", "ArAiEval"])
    args = parser.parse_args()
    args.no_shots = int(args.no_shots)

    df = pd.read_csv(f"./datasets/{args.task}/{args.input}")

    prompt_template = None
    if args.prompt:
        with open(args.prompt, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    params = {
        "provider": args.provider,
        "model": args.model
    }

    acc, prec, rec, f1, preds = calculate_metrics(df, params, template=prompt_template, no_shots=args.no_shots)

    print("\n--- Classification Report ---")
    print(f"Accuracy : {round(acc, 3)}")
    print(f"Precision: {round(prec, 3)}")
    print(f"Recall   : {round(rec, 3)}")
    print(f"F1 Score : {round(f1, 3)}")

    if args.output is None:
        os.makedirs(f'./predictions/{args.provider}_{args.model}/{args.task}', exist_ok=True)
        args.output = f"./predictions/{args.task}/pred_s{args.no_shots}{'_cot' if args.prompt == 'prompt_cot.txt' else ''}.csv"
    df["Predicted"] = preds
    df.to_csv(args.output, index=False)

    # Append classification report to the same file
    with open(args.output, "a", encoding="utf-8") as f:
        f.write("\n--- Classification Report ---\n")
        f.write(f"Accuracy , {round(acc, 3)}\n")
        f.write(f"Precision, {round(prec, 3)}\n")
        f.write(f"Recall   , {round(rec, 3)}\n")
        f.write(f"F1 Score , {round(f1, 3)}\n")
