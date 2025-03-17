import warnings; warnings.filterwarnings("ignore")
from src.utils.config import Config
from src.embeddings.embedding_handler import EmbeddingHandler
from src.embeddings.vector_store import VectorStore
from src.generation.llm_handler import LLMHandler
from src.generation.prompt_handler import PromptHandler
from src.utils.log_handler import LogHandler


class MultiModalRAG:



    def __init__(self):
        self.config = Config()
        self.embedding_handler = EmbeddingHandler(self.config)
        self.vector_store = VectorStore(self.config)
        self.llm_handler = LLMHandler(self.config)
        self.log_handler = LogHandler("logs")
        
    def process_query(self, query_text):
        # Get text embeddings and search
        # print("Text Search")
        text_embedding = self.embedding_handler.get_text_embeddings(query_text)
        text_results = self.vector_store.query_vectors(
            text_embedding, 
            self.config.top_k,
            index_type="text"
        )
        
        # # Get image embeddings and search
        # print("Image Search")
        # image_embedding = self.embedding_handler.get_clip_text_embeddings(query_text)
        # image_results = self.vector_store.query_vectors(
        #     image_embedding, 
        #     self.config.top_k,
        #     index_type="image"
        # )
        
        # Convert Pinecone QueryResponse to dict for JSON serialization
        text_results_dict = {
            "matches": [
                {
                    "id": str(match.id),
                    "score": float(match.score),
                    "metadata": dict(match.metadata) if match.metadata else {},
                    "values": match.values if hasattr(match, 'values') else []
                }
                for match in text_results.matches
            ],
            "namespace": text_results.namespace if hasattr(text_results, 'namespace') else "",
            "usage": text_results.usage.to_dict() if hasattr(text_results, 'usage') else {}
        }
        
        # Format retrieved documents
        # print("Formatting Results")
        retrieved_docs = self._format_results(text_results)
        # image_descriptions = self._format_results(image_results)
        
        # Create prompt and generate response
        # print("Creating Prompt")
        prompt = PromptHandler.create_rag_prompt(
            query_text, 
            retrieved_docs, 
            image_descriptions=[]
        )
        
        response = self.llm_handler.generate_response(prompt)

        self.log_handler.save_interaction(
            query=query_text,
            text_results=text_results,
            retrieved_docs=retrieved_docs,
            response=response,
            config=self.config
        )

        return response
    
    def _format_results(self, results):
        return "\n\n".join(
            f"Score: {match['score']}\nContent: {match['metadata'].get('content', 'No content')}"
            for match in results["matches"]
        )
