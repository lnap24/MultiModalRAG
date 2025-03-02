from pinecone import Pinecone, ServerlessSpec

class VectorStore:
    def __init__(self, config):
        print("\nConnecting to Pinecone...", end='', flush=True)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        print("DONE")
        self.setup_indices(config)
        
        
    def setup_indices(self, config):
        print("Setting up Pinecone indices...", end='', flush=True)
        # Setup text index
        if config.index_name_text not in self.pc.list_indexes().names():
            print(f"\nCreating new text index: {config.index_name_text}...", end='', flush=True)
            self.pc.create_index(
                name=config.index_name_text,
                dimension=config.text_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("DONE")
            
        # Setup image index
        if config.index_name_image not in self.pc.list_indexes().names():
            print(f"\nCreating new image index: {config.index_name_image}...", end='', flush=True)
            self.pc.create_index(
                name=config.index_name_image,
                dimension=config.image_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("DONE")
            
        self.text_index = self.pc.Index(config.index_name_text)
        self.image_index = self.pc.Index(config.index_name_image)
        print("DONE")
        
    def store_vectors(self, vectors, metadata=None, index_type="text"):
        print(f"Storing vectors in Pinecone {index_type} index...", end='', flush=True)
        index = self.text_index if index_type == "text" else self.image_index
        result = index.upsert(vectors=vectors)
        print("DONE")
        return result
        
    def query_vectors(self, query_vector, top_k=5, index_type="text"):
        print(f"Querying Pinecone {index_type} index for top {top_k} matches...", end='', flush=True)
        index = self.text_index if index_type == "text" else self.image_index
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        print("DONE")
        return result 