from rouge_score import rouge_scorer
from bert_score import score
import numpy as np

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge_bert_metrics_batch(examples):
    """
    Compute ROUGE and BERTScore for a batch of examples.
    
    Args:
        examples: A dictionary containing lists of "human_preferred" and "model_response".

    Returns:
        A dictionary with lists of ROUGE and BERTScore metrics.
    """
    human_texts = examples["human_preferred"]
    model_texts = examples["model_response"]

    rouge_1_scores, rouge_2_scores, rouge_L_scores, bert_scores = [], [], [], []

    # Compute ROUGE scores for each pair
    for human_text, model_text in zip(human_texts, model_texts):
        rouge_scores = rouge.score(human_text, model_text)
        rouge_1_scores.append(rouge_scores["rouge1"].fmeasure)
        rouge_2_scores.append(rouge_scores["rouge2"].fmeasure)
        rouge_L_scores.append(rouge_scores["rougeL"].fmeasure)

    # Compute BERTScore for the entire batch at once
    P, R, F1 = score(model_texts, human_texts, lang="en")

    return {
        "rouge-1": rouge_1_scores,
        "rouge-2": rouge_2_scores,
        "rouge-L": rouge_L_scores,
        "bert-score": F1.tolist(),  # Convert tensor to list
    }