import json
import os
from datetime import datetime

class LogHandler:
    def __init__(self, log_dir="logs"):
        """Initialize log handler with directory path"""
        print("\nInitializing log handler...", end='', flush=True)
        self.log_dir = log_dir
        self._ensure_log_directory()
        self.log_file = os.path.join(self.log_dir, "interactions.json")
        self._initialize_log_file()
        print("DONE")
        
    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _initialize_log_file(self):
        """Initialize the log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"interactions": []}, f, indent=2)
            
    def _format_text_results(self, text_results):
        """Format text results into proper JSON structure"""
        try:
            # Convert Pinecone QueryResponse to dict
            if hasattr(text_results, 'to_dict'):
                formatted_results = text_results.to_dict()
            elif isinstance(text_results, dict):
                formatted_results = text_results
            elif isinstance(text_results, str):
                formatted_results = eval(text_results)
            else:
                # Convert the object to a dictionary format we want
                formatted_results = {
                    "matches": [
                        {
                            "id": match.id if hasattr(match, 'id') else None,
                            "score": match.score if hasattr(match, 'score') else None,
                            "metadata": match.metadata if hasattr(match, 'metadata') else {},
                            "values": match.values if hasattr(match, 'values') else []
                        }
                        for match in text_results.matches
                    ],
                    "namespace": getattr(text_results, 'namespace', ''),
                    "usage": text_results.usage.to_dict() if hasattr(text_results, 'usage') else {}
                }
            
            # Clean up the metadata text format
            for match in formatted_results.get('matches', []):
                if 'metadata' in match and 'text' in match['metadata']:
                    match['metadata']['text'] = ' '.join(
                        match['metadata']['text'].replace('\n', ' ').split()
                    )
            
            return formatted_results
        except Exception as e:
            print(f"\nWarning: Error formatting text results: {str(e)}")
            # Return a simplified version of the results
            try:
                return {
                    "matches": [
                        {
                            "id": str(match.id) if hasattr(match, 'id') else "unknown",
                            "score": float(match.score) if hasattr(match, 'score') else 0.0,
                            "metadata": dict(match.metadata) if hasattr(match, 'metadata') else {},
                        }
                        for match in text_results.matches
                    ]
                }
            except:
                return {"error": "Could not format text results", "type": str(type(text_results))}
            
    def save_interaction(self, query, retrieved_docs, text_results, response, config):
        """Append interaction details to the JSON file"""
        print("Saving interaction to log...", end='', flush=True)
        
        # Get model details from config
        model_provider = config.llm_provider
        model_name = config.ollama_model if model_provider == "ollama" else config.openai_model
        
        # Format text results properly
        formatted_text_results = self._format_text_results(text_results)
        
        # Prepare new interaction data
        new_interaction = {
            "timestamp": datetime.now().isoformat(),
            "model_provider": model_provider,
            "model_name": model_name,
            "query": query,
            "retrieved_documents": retrieved_docs,
            "text_results": formatted_text_results,
            "model_response": response
        }
        
        # Read existing data
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {"interactions": []}
        
        # Append new interaction
        data["interactions"].append(new_interaction)
        
        # Write back to file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print("DONE") 