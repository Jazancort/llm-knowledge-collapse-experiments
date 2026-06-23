import re
import numpy as np
import torch


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truths: list[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(gt) == pred_norm for gt in ground_truths)


def batch_accuracy(predictions: list[str], ground_truths: list[list[str]]) -> float:
    correct = sum(exact_match(p, gts) for p, gts in zip(predictions, ground_truths))
    return correct / len(predictions) if predictions else 0.0


def token_log_probs(scores: tuple, generated_ids: torch.Tensor) -> dict[str, float]:
    """Compute confidence metrics from generation scores.

    Args:
        scores: tuple of (vocab_size,) logit tensors per generated token
        generated_ids: tensor of generated token ids
    """
    log_probs = []
    entropies = []

    for i, logits in enumerate(scores):
        probs = torch.softmax(logits[0].float(), dim=-1)
        log_p = torch.log_softmax(logits[0].float(), dim=-1)

        token_id = generated_ids[i].item()
        log_probs.append(log_p[token_id].item())

        entropy = -(probs * log_p).sum().item()
        entropies.append(entropy)

    return {
        "avg_log_prob": float(np.mean(log_probs)),
        "avg_entropy": float(np.mean(entropies)),
    }


def distinct_n(texts: list[str], n: int = 4) -> float:
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
