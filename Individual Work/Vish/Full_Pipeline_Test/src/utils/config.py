from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        
        # API Keys
        self.huggingface_api_key = os.getenv("hugging_face_key")
        self.pinecone_api_key = os.getenv("pinecone_api_key")
        self.mongo_uri = os.getenv("mongo_db_key")
        self.openai_api_key = os.getenv("open_ai_api_key")
        
        # Model Settings
        self.text_model_id = "sentence-transformers/all-mpnet-base-v2"  # Text embeddings model
        self.clip_model_id = "openai/clip-vit-base-patch32"  # Image embeddings model
        
        # LLM Settings
        self.llm_provider = "ollama"  # Default Method
        self.openai_model = "gpt-4"  # OpenAI model
        self.ollama_model = "phi"  # Ollama model
        
        # Vector Store Settings
        self.index_name_text = "rag-app"  # Separate index for text
        self.index_name_image = "rag-app"  # Separate index for images
        self.text_dimension = 768  # all-mpnet-base-v2 dimension
        self.image_dimension = 512  # CLIP dimension
        
        # Data Processing Settings
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.top_k = 5 