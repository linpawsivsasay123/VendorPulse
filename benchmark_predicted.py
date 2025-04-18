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
    # Check common fields: category, type, question
    if not isinstance(item.get('category'), str) or not item['category'].strip():
        return False, "Invalid or missing category"
    if not isinstance(item.get('type'), str) or not item['type'].strip():
        return False, "Invalid or missing type"
    if not isinstance(item.get('question'), str) or not item['question'].strip():
        return False, "Invalid or missing question"

    # Validate type-specific fields
    question_type = item['type'].lower()
    if question_type in ['compliance', 'other']:
        # Check weight field
        if 'weight' not in item or not isinstance(item['weight'], (int, float)):
            return False, "Missing or invalid weight for compliance/other type"
        if not (0 <= item['weight'] <= 10):
            return False, "Weight out of range (0-10) for compliance/other type"
    elif question_type == 'external_surface':
        # Check weights_vulnerability_weight
        if 'weights_vulnerability_weight' not in item or not isinstance(item['weights_vulnerability_weight'], (int, float)):
            return False, "Missing or invalid weights_vulnerability_weight for external_surface type"
        if not (0 <= item['weights_vulnerability_weight'] <= 10):
            return False, "weights_vulnerability_weight out of range (0-10) for external_surface type"
        # Check weights_likelihood_probability
        if 'weights_likelihood_probability' not in item or not isinstance(item['weights_likelihood_probability'], (int, float)):
            return False, "Missing or invalid weights_likelihood_probability for external_surface type"
        if not (0 <= item['weights_likelihood_probability'] <= 1):
            return False, "weights_likelihood_probability out of range (0-1) for external_surface type"
        # Check weights_asset_impact
        if 'weights_asset_impact' not in item or not isinstance(item['weights_asset_impact'], (int, float)):
            return False, "Missing or invalid weights_asset_impact for external_surface type"
        if not (0 <= item['weights_asset_impact'] <= 1):
            return False, "weights_asset_impact out of range (0-1) for external_surface type"
    else:
        return False, f"Unknown question type: {question_type}"

    return True, "Valid question"
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts."""
    # Tokenize and clean texts
    tokens1 = set(word_tokenize(text1.lower())) - stop_words - punctuation
    tokens2 = set(word_tokenize(text2.lower())) - stop_words - punctuation
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def compute_scores(data):
    """Compute aggregated similarity scores across all valid questions."""
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

        # Encode generated question for semantic similarity
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

    # Compute aggregated results
    result = {
        'semantic_similarity_mean': float(np.mean(semantic_scores)) if semantic_scores else 0.0,
        'semantic_similarity_std': float(np.std(semantic_scores)) if semantic_scores else 0.0,
        'jaccard_similarity': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        'rouge1_f1': float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        'rouge2_f1': float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
        'rougeL_f1': float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        'bleu_score': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        'total_comparisons': len(semantic_scores)
    }

    return result

def append_to_json(data, result, output_file='vendor_similarity_scores.json'):
    """Append vendor results to a JSON file with specified similarity scores."""
    vendor_result = {
        "vendor_metadata": {
            "sector": data["vendor_metadata_sector"],
            "domain": data["vendor_metadata_domain"],
            "location": data["vendor_metadata_location"],
            "employee_strength": data["vendor_metadata_employee_strength"]
        },
        "similarity_scores": {
            "semantic_similarity_mean": result["semantic_similarity_mean"],
            "semantic_similarity_std": result["semantic_similarity_std"],
            "jaccard_similarity": result["jaccard_similarity"],
            "rouge1_f1": result["rouge1_f1"],
            "rouge2_f1": result["rouge2_f1"],
            "rougeL_f1": result["rougeL_f1"],
            "bleu_score": result["bleu_score"],
            "total_comparisons": result["total_comparisons"]
        }
    }

    # Load existing JSON file or initialize a new one
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                json_data = {"vendors": []}
    else:
        json_data = {"vendors": []}

    # Append the new vendor result
    json_data["vendors"].append(vendor_result)

    # Write back to the file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

def main():
    
    with open("./predicted_weights.json", "r") as f:
        input_data = json.load(f)
        
    for data in input_data:
    # Load the dictionary (replace with actual dictionary or file input)
        # Compute scores
        
        valid_data = {
            "vendor_metadata_sector": data["vendor_metadata_sector"],
            "vendor_metadata_domain": data["vendor_metadata_domain"],
            "vendor_metadata_location": data["vendor_metadata_location"],
            "vendor_metadata_employee_strength": data["vendor_metadata_employee_strength"],
            "questions": []
        }

        # Filter valid questions
        for item in data['questions']:
            valid, _ = check_valid_question(item)
            if valid:
                valid_data['questions'].append(item)

        # Sample up to 10 valid questions
        sampled_questions = random.sample(valid_data['questions'], min(10, len(valid_data['questions'])))
        valid_data['questions'] = sampled_questions

        # Compute scores
        result = compute_scores(valid_data)

        append_to_json(valid_data,result)
        # Print summary
        # print("Aggregated Similarity Scores Across All Questions:")
        # print(f"  Semantic Similarity Mean: {result['semantic_similarity_mean']:.4f}")
        # print(f"  Semantic Similarity Std: {result['semantic_similarity_std']:.4f}")
        # print(f"  Jaccard Similarity: {result['jaccard_similarity']:.4f}")
        # print(f"  ROUGE-1 F1: {result['rouge1_f1']:.4f}")
        # print(f"  ROUGE-2 F1: {result['rouge2_f1']:.4f}")
        # print(f"  ROUGE-L F1: {result['rougeL_f1']:.4f}")
        # print(f"  BLEU Score: {result['bleu_score']:.4f}")
        # print(f"  Total Comparisons: {result['total_comparisons']}")

if __name__ == "__main__":
    main()
    


    