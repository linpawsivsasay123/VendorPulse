import torch
import json
from sentence_transformers import SentenceTransformer, util

class InferenceModel_Similarity:
    
    def __init__(self):
        self.MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.MODEL_DIR = "./Model/Model_similarity"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Similarity Using device: {self.device}")
        
        self.model = SentenceTransformer(self.MODEL_NAME, cache_folder=self.MODEL_DIR)
        self.dataset_path = "./datasets/questions_with_weights.json"
        print("Similarity Model is on", self.device)
    
    def most_similar(self, Question, question_tag, top_n=5):
        """Finds the top_n most similar questions and returns their corresponding dictionaries."""

        # Load dataset
        with open(self.dataset_path, "r") as file:
            data = json.load(file)  
        
        filtered_data = [item for item in data if item.get("type") == question_tag]

        if not filtered_data:
            return []  

        all_questions = [item["question"] for item in filtered_data]

        query_embedding = self.model.encode([Question], convert_to_tensor=True)
        question_embeddings = self.model.encode(all_questions, convert_to_tensor=True)


        similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

        similarity_scores = [(float(similarities[idx]), idx) for idx in range(len(all_questions))]

        similarity_scores.sort(reverse=True, key=lambda x: x[0])

        top_items = [filtered_data[idx] for _, idx in similarity_scores[:top_n]]

        return top_items  
