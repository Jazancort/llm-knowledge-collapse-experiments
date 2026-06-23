"""M1A Step 3 (corrected): Accuracy, Confidence & Fluency Baseline.

Fixes from review:
1. Uses tokenizer.apply_chat_template (model's native format)
2. Entropy via Categorical distribution (numerically stable, no NaN)
3. Number words normalized in exact match

Run: uv run python scripts/m1a_baseline.py
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROBE_SET = [
    {"question": "What is the capital of France?", "answers": ["Paris"]},
    {"question": "Who wrote Romeo and Juliet?", "answers": ["William Shakespeare", "Shakespeare"]},
    {"question": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"question": "What planet is closest to the Sun?", "answers": ["Mercury"]},
    {"question": "Who painted the Mona Lisa?", "answers": ["Leonardo da Vinci", "Da Vinci", "Leonardo"]},
    {"question": "What is the largest ocean on Earth?", "answers": ["Pacific Ocean", "Pacific"]},
    {"question": "What year did World War II end?", "answers": ["1945"]},
    {"question": "Who developed the theory of relativity?", "answers": ["Albert Einstein", "Einstein"]},
    {"question": "What is the smallest prime number?", "answers": ["2", "two"]},
    {"question": "What is the capital of Japan?", "answers": ["Tokyo"]},
    {"question": "Who invented the telephone?", "answers": ["Alexander Graham Bell", "Bell"]},
    {"question": "What is the hardest natural substance?", "answers": ["Diamond"]},
    {"question": "What gas do plants absorb from the atmosphere?", "answers": ["Carbon dioxide", "CO2"]},
    {"question": "Who was the first person to walk on the Moon?", "answers": ["Neil Armstrong", "Armstrong"]},
    {"question": "What is the capital of Australia?", "answers": ["Canberra"]},
    {"question": "What is the largest mammal?", "answers": ["Blue whale"]},
    {"question": "In what year was the Berlin Wall torn down?", "answers": ["1989"]},
    {"question": "What element has atomic number 1?", "answers": ["Hydrogen"]},
    {"question": "Who wrote 1984?", "answers": ["George Orwell", "Orwell"]},
    {"question": "What is the boiling point of water in Celsius?", "answers": ["100", "100 degrees"]},
]

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truths: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(normalize_answer(gt) in pred or pred in normalize_answer(gt) for gt in ground_truths)


def distinct_n(texts: list[str], n: int = 4) -> float:
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def main():
    print("=" * 60)
    print("M1A STEP 3: ACCURACY & CONFIDENCE BASELINE (CORRECTED)")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\n[1/3] Loading {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model.eval()

    # Generate answers using proper chat template
    print(f"\n[2/3] Generating answers ({len(PROBE_SET)} questions, chat template)...")

    predictions = []
    all_log_probs = []
    all_entropies = []

    for item in PROBE_SET:
        messages = [
            {"role": "system", "content": "Answer the following question in 5 words or less."},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(text)

        # Confidence: avg log-prob of generated tokens
        log_probs = []
        entropies = []
        for i, logits in enumerate(outputs.scores):
            if i >= len(generated_ids):
                break
            logits_f32 = logits[0].float()
            token_id = generated_ids[i].item()

            # Log-prob of chosen token
            log_p = torch.log_softmax(logits_f32, dim=-1)
            log_probs.append(log_p[token_id].item())

            # Entropy via Categorical (numerically stable)
            ent = Categorical(logits=logits_f32).entropy().item()
            entropies.append(ent)

        if log_probs:
            all_log_probs.append(np.mean(log_probs))
        if entropies:
            all_entropies.append(np.mean(entropies))

    # Evaluate
    print(f"\n[3/3] Results:")
    correct = 0
    print(f"\n  {'#':<4} {'OK':<4} {'Prediction':<30} {'Ground Truth'}")
    print(f"  {'-'*68}")
    for i, (pred, item) in enumerate(zip(predictions, PROBE_SET)):
        match = exact_match(pred, item["answers"])
        if match:
            correct += 1
        print(f"  {i:<4} {'Y' if match else 'N':<4} {pred[:28]:<30} {item['answers'][0]}")

    accuracy = correct / len(PROBE_SET)
    avg_log_prob = np.mean(all_log_probs) if all_log_probs else 0
    avg_entropy = np.mean(all_entropies) if all_entropies else 0
    d4 = distinct_n(predictions, n=4)

    print(f"\n  {'='*40}")
    print(f"  BASELINE METRICS (Gen 0, with chat template)")
    print(f"  {'='*40}")
    print(f"  Accuracy:       {accuracy:.2%} ({correct}/{len(PROBE_SET)})")
    print(f"  Avg log-prob:   {avg_log_prob:.4f}")
    print(f"  Avg entropy:    {avg_entropy:.4f}")
    print(f"  Distinct-4:     {d4:.4f}")
    print(f"  Entropy NaN:    {'YES - PROBLEM' if np.isnan(avg_entropy) else 'NO - CLEAN'}")

    if accuracy >= 0.5:
        print(f"\n  Baseline adequate for collapse observation.")
    else:
        print(f"\n  WARNING: Baseline too low.")

    print(f"\n  M1A Step 3 (corrected): COMPLETE")


if __name__ == "__main__":
    main()
