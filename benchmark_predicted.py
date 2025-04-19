from LLM.inference_similarity import InferenceModel_Similarity
import json
import re
import matplotlib.pyplot as plt
import uuid
import os
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')
obj = InferenceModel_Similarity()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
smoother = SmoothingFunction()

def check_valid_question(item):
    """Validate a question based on category, type, and weight fields."""
    if not isinstance(item.get('category'), str) or not item['category'].strip():
        return False, "Invalid or missing category"
    if not isinstance(item.get('type'), str) or not item['type'].strip():
        return False, "Invalid or missing type"
    if not isinstance(item.get('question'), str) or not item['question'].strip():
        return False, "Invalid or missing question"

    question_type = item['type'].lower()
    if question_type in ['compliance', 'other']:
        if 'weight' not in item or not isinstance(item['weight'], (int, float)):
            return False, "Missing or invalid weight for compliance/other type"
        if not (0 <= item['weight'] <= 10):
            return False, "Weight out of range (0-10) for compliance/other type"
    elif question_type == 'external_surface':
        if 'weights_vulnerability_weight' not in item or not isinstance(item['weights_vulnerability_weight'], (int, float)):
            return False, "Missing or invalid weights_vulnerability_weight for external_surface type"
        if not (0 <= item['weights_vulnerability_weight'] <= 10):
            return False, "weights_vulnerability_weight out of range (0-10) for external_surface type"
        if 'weights_likelihood_probability' not in item or not isinstance(item['weights_likelihood_probability'], (int, float)):
            return False, "Missing or invalid weights_likelihood_probability for external_surface type"
        if not (0 <= item['weights_likelihood_probability'] <= 1):
            return False, "weights_likelihood_probability out of range (0-1) for external_surface type"
        if 'weights_asset_impact' not in item or not isinstance(item['weights_asset_impact'], (int, float)):
            return False, "Missing or invalid weights_asset_impact for external_surface type"
        if not (0 <= item['weights_asset_impact'] <= 1):
            return False, "weights_asset_impact out of range (0-1) for external_surface type"
    else:
        return False, f"Unknown question type: {question_type}"

    return True, "Valid question"

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts."""
    tokens1 = set(word_tokenize(text1.lower())) - stop_words - punctuation
    tokens2 = set(word_tokenize(text2.lower())) - stop_words - punctuation
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def compute_scores(data):
    """Compute aggregated similarity scores and variances across all valid questions."""
    semantic_scores = []
    jaccard_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for idx, item in enumerate(data['questions']):
        generated_question = item['question']
        similar_questions = item.get('similar', [])
        if not similar_questions:
            continue

        generated_embedding = obj.model.encode(generated_question, convert_to_tensor=True)

        for similar in similar_questions:
            similar_question = similar['question']

            # Semantic similarity
            similar_embedding = obj.model.encode(similar_question, convert_to_tensor=True)
            semantic_score = util.cos_sim(generated_embedding, similar_embedding).item()
            semantic_scores.append(semantic_score)

            # Jaccard similarity
            jaccard_score = jaccard_similarity(generated_question, similar_question)
            jaccard_scores.append(jaccard_score)

            # ROUGE scores
            rouge_scores = scorer.score(generated_question, similar_question)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

            # BLEU score
            reference = [word_tokenize(similar_question.lower())]
            candidate = word_tokenize(generated_question.lower())
            bleu_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
            bleu_scores.append(bleu_score)

    result = {
        'semantic_similarity_mean': float(np.mean(semantic_scores)) if semantic_scores else 0.0,
        'semantic_similarity_std': float(np.std(semantic_scores)) if semantic_scores else 0.0,
        'jaccard_similarity': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        'jaccard_variance': float(np.var(jaccard_scores)) if jaccard_scores else 0.0,
        'rouge1_f1': float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        'rouge1_variance': float(np.var(rouge1_scores)) if rouge1_scores else 0.0,
        'rouge2_f1': float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
        'rouge2_variance': float(np.var(rouge2_scores)) if rouge2_scores else 0.0,
        'rougeL_f1': float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        'rougeL_variance': float(np.var(rougeL_scores)) if rougeL_scores else 0.0,
        'bleu_score': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        'bleu_variance': float(np.var(bleu_scores)) if bleu_scores else 0.0,
        'total_comparisons': len(semantic_scores)
    }

    return result

def append_to_json(group_data, result, output_file='vendor_similarity_scores.json'):
    """Append group results to a JSON file with specified similarity scores and variances."""
    group_result = {
        "sectors": group_data["sectors"],
        "vendors": [
            {
                "sector": data["vendor_metadata_sector"],
                "domain": data["vendor_metadata_domain"],
                "location": data["vendor_metadata_location"],
                "employee_strength": data["vendor_metadata_employee_strength"]
            } for data in group_data["vendors"]
        ],
        "similarity_scores": {
            "semantic_similarity_mean": result["semantic_similarity_mean"],
            "semantic_similarity_std": result["semantic_similarity_std"],
            "jaccard_similarity": result["jaccard_similarity"],
            "jaccard_variance": result["jaccard_variance"],
            "rouge1_f1": result["rouge1_f1"],
            "rouge1_variance": result["rouge1_variance"],
            "rouge2_f1": result["rouge2_f1"],
            "rouge2_variance": result["rouge2_variance"],
            "rougeL_f1": result["rougeL_f1"],
            "rougeL_variance": result["rougeL_variance"],
            "bleu_score": result["bleu_score"],
            "bleu_variance": result["bleu_variance"],
            "total_comparisons": result["total_comparisons"]
        }
    }

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                json_data = json.load(f)
                # Ensure 'groups' key exists
                if "groups" not in json_data:
                    print(f"Warning: Existing file {output_file} has unexpected structure. Initializing with empty groups.")
                    json_data = {"groups": []}
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {output_file}. Initializing with empty groups.")
                json_data = {"groups": []}
    else:
        json_data = {"groups": []}

    json_data["groups"].append(group_result)

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

def main():
    # Define sector_groups list of lists (sectors to group together)
    sector_groups = [
        ["Financial Services"],
        ["IT Services", "Security Services"],
        ["Legal Services"],
        ["Retail"],
        ["Service-Based"]
    ]

    # Load input data
    with open("./predicted_weights.json", "r") as f:
        input_data = json.load(f)

    # Process each group of sectors
    for group_idx, sectors in enumerate(sector_groups, 1):
        print(f"\nProcessing group {group_idx}: {', '.join(sectors)}")

        # Collect vendors matching the sectors
        group_data = {
            "sectors": sectors,
            "vendors": [],
            "questions": []
        }

        for data in input_data:
            if data["vendor_metadata_sector"] in sectors:
                group_data["vendors"].append(data)
                # Filter valid questions
                for item in data['questions']:
                    valid, _ = check_valid_question(item)
                    if valid:
                        group_data['questions'].append(item)

        if not group_data["vendors"]:
            print(f"Warning: No vendors found for sectors {sectors}")
            continue

        if not group_data["questions"]:
            print(f"Warning: No valid questions found for sectors {sectors}")
            continue

        # Sample up to 10 valid questions
        sampled_questions = random.sample(group_data['questions'], min(10, len(group_data['questions'])))
        group_data['questions'] = sampled_questions

        # Compute scores for the group
        result = compute_scores(group_data)

        # Append to JSON
        append_to_json(group_data, result)

if __name__ == "__main__":
    main()