from openai import OpenAI
import ollama

class LLMHandler:
    def __init__(self, config):
        self.provider = config.llm_provider
        
        if self.provider == "openai":
            print("\nInitializing OpenAI client...", end='', flush=True)
            self.client = OpenAI(api_key=config.openai_api_key)
            self.model = config.openai_model
            print("DONE")
        else:  # ollama
            print("\nInitializing Ollama client...", end='', flush=True)
            self.model = config.ollama_model 
            print("DONE")

    def generate_response(self, prompt, images=None):
        
        if self.provider == "openai":
            return self._generate_openai_response(query, prompt)
        else:
            return self._generate_ollama_response(prompt)
            
    def _generate_openai_response(self, prompt):
        print("Generating OpenAI response...", end='', flush=True)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specializing in computer science and machine learning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            result = response.choices[0].message.content
            print("DONE")
            return result
        except Exception as e:
            print(f"\nError generating OpenAI response: {str(e)}")
            return "Error: Failed to generate response"
            
    def _generate_ollama_response(self, prompt):
        print("Generating Ollama response...", end='', flush=True)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            result = response['message']['content']
            print("DONE")
            return result
        except Exception as e:
            print(f"\nError generating Ollama response: {str(e)}")
            return "Error: Failed to generate response" 