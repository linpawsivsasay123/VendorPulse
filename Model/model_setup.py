
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
# Login to Hugging Face
hf_token = "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"
login(token=hf_token)

# Define model directory
MODEL_NAME_GENERATE = "meta-llama/Llama-2-7b-chat-hf"
MODEL_DIR_GENERATE = "./Model_generate"  

MODEL_NAME_SIMILARITY = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR_SIMILARITY = "./Model_similarity"  

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GENERATE, cache_dir=MODEL_DIR_GENERATE)

# Load Model in 4-bit Quantization for Faster Inference
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_GENERATE,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=MODEL_DIR_GENERATE
)



model_ = SentenceTransformer(MODEL_NAME_SIMILARITY, cache_folder=MODEL_DIR_SIMILARITY)