import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import sys

class EmbeddingHandler:
    def __init__(self, config):
        print("\nInitializing embedding models...", end='', flush=True)
        # CLIP for image embeddings
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_id)
        
        # SentenceTransformer for text embeddings
        self.text_model = SentenceTransformer(config.text_model_id)
        print("DONE")
    
    def get_image_embeddings(self, image):
        print("Processing image embeddings...", end='', flush=True)
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model.get_image_features(**inputs)
        result = outputs.detach().numpy().tolist()[0]
        print("DONE")
        return result
    
    def get_text_embeddings(self, text):
        print("Generating text embeddings...", end='', flush=True)
        result = self.text_model.encode(text)
        print("DONE")
        return result.tolist()  # Convert numpy array to list
    
    def get_clip_text_embeddings(self, text):
        print("Generating CLIP text embeddings for image search...", end='', flush=True)
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.clip_model.get_text_features(**inputs)
        result = outputs.detach().numpy().tolist()[0]
        print("DONE")
        return result 