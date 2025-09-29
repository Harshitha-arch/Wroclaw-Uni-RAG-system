import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class SemanticSearchEngine:
    def __init__(self, knowledge_base_path="semantic_knowledge_base.json", model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the semantic search engine
        
        Args:
            knowledge_base_path: Path to the JSON knowledge base
            model_name: Name of the sentence transformer model (should match chunking model)
        """
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        
        # Load the same model used for chunking
        print(f"🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load knowledge base
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load and prepare the knowledge base"""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"❌ Knowledge base not found: {self.knowledge_base_path}")
        
        print(f"📚 Loading knowledge base from: {self.knowledge_base_path}")
        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old format (direct list) and new format (with metadata)
        if isinstance(data, list):
            self.chunks = data
        elif isinstance(data, dict) and "chunks" in data:
            self.chunks = data["chunks"]
            self.metadata = data.get("metadata", {})
        else:
            raise ValueError("❌ Invalid knowledge base format")
        
        # Convert embeddings to numpy array
        self.embeddings = np.array([chunk["embedding"] for chunk in self.chunks])
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        print(f"📊 Embedding dimensions: {self.embeddings.shape[1]}")
    
    def search(self, query, top_k=5, min_similarity=0.1):
        """
        Search for relevant chunks
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of dictionaries with chunk info and similarity scores
        """
        if not query.strip():
            return []
        
        # Generate query embedding
        print(f"🔍 Searching for: '{query}'")
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by minimum similarity and prepare results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= min_similarity:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.get("id", f"chunk_{idx}"),
                    "content": chunk.get("content", ""),
                    "similarity": float(similarity),
                    "cluster": chunk.get("cluster", -1),
                    "word_count": chunk.get("word_count", len(chunk.get("content", "").split())),
                    "content_length": chunk.get("content_length", len(chunk.get("content", "")))
                })
        
        return results
    
    def print_results(self, results, show_full_content=True):
        """Pretty print search results"""
        if not results:
            print("❌ No relevant chunks found.")
            return
        
        print(f"\n🎯 Found {len(results)} relevant chunks:")
        print("=" * 100)
        
        for i, result in enumerate(results, 1):
            chunk_id = result["chunk_id"]
            similarity = result["similarity"]
            cluster = result["cluster"]
            word_count = result["word_count"]
            
            # Extract chunk number for display
            chunk_number = chunk_id.split("_")[-1].lstrip("0") or "0"
            
            print(f"\n📝 Rank {i} | Chunk #{chunk_number} | Similarity: {similarity:.4f} | Cluster: {cluster} | Words: {word_count}")
            print("-" * 100)
            
            if show_full_content:
                content = result["content"]
                # Truncate very long content for better readability
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                print(content)
            else:
                # Show just first 200 characters
                content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                print(content_preview)
            
            print("-" * 100)
    
    def interactive_search(self):
        """Start interactive search session"""
        print("📚 Enhanced Semantic Search Engine")
        print("💡 Tips:")
        print("   - Type 'exit' or 'quit' to stop")
        print("   - Type 'help' for commands")
        print("   - Ask questions in natural language")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n🔍 Enter your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif query.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                
                elif query.lower().startswith('config'):
                    self.show_config()
                    continue
                
                elif query.lower().startswith('stats'):
                    self.show_stats()
                    continue
                
                elif not query:
                    print("⚠️ Please enter a question.")
                    continue
                
                # Perform search
                results = self.search(query, top_k=5, min_similarity=0.1)
                
                # Show results
                if results:
                    self.print_results(results, show_full_content=True)
                    
                    # Show similarity distribution
                    similarities = [r["similarity"] for r in results]
                    print(f"\n📊 Similarity Range: {min(similarities):.4f} - {max(similarities):.4f}")
                    
                    # Ask if user wants to see more results
                    if len(results) == 5:
                        more = input("\n❓ Show more results? (y/n): ").strip().lower()
                        if more in ['y', 'yes']:
                            extended_results = self.search(query, top_k=10, min_similarity=0.05)
                            if len(extended_results) > 5:
                                print("\n📋 Additional Results:")
                                self.print_results(extended_results[5:], show_full_content=False)
                else:
                    print("❌ No relevant chunks found. Try:")
                    print("   - Using different keywords")
                    print("   - Being more specific")
                    print("   - Checking for spelling errors")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 Available Commands:")
        print("   help     - Show this help message")
        print("   config   - Show current configuration")
        print("   stats    - Show knowledge base statistics")
        print("   exit/quit - Exit the program")
        print("\n💡 Search Tips:")
        print("   - Ask questions in natural language")
        print("   - Use specific keywords from your domain")
        print("   - Try different phrasings if no results found")
    
    def show_config(self):
        """Show current configuration"""
        print(f"\n⚙️ Current Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Knowledge Base: {self.knowledge_base_path}")
        print(f"   Total Chunks: {len(self.chunks)}")
        print(f"   Embedding Dimensions: {self.embeddings.shape[1]}")
    
    def show_stats(self):
        """Show knowledge base statistics"""
        if not hasattr(self, 'metadata'):
            print("📊 Basic Statistics:")
            print(f"   Total Chunks: {len(self.chunks)}")
            return
        
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"   Total Chunks: {self.metadata.get('total_chunks', len(self.chunks))}")
        print(f"   Embedding Model: {self.metadata.get('embedding_model', 'Unknown')}")
        print(f"   Created: {self.metadata.get('creation_timestamp', 'Unknown')}")
        
        if 'cluster_distribution' in self.metadata:
            print(f"   Cluster Distribution:")
            for cluster, count in self.metadata['cluster_distribution'].items():
                print(f"     {cluster}: {count} chunks")

def main():
    """Main function to run the search engine"""
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            knowledge_base_path="semantic_knowledge_base.json",
            model_name="paraphrase-multilingual-MiniLM-L12-v2"  # Same model as used in chunking
        )
        
        # Start interactive search
        search_engine.interactive_search()
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure to run semantic_chunking.py first to create the knowledge base.")
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")

if __name__ == "__main__":
    main()

"""
This script implements a Semantic Search Engine that:

1. Loads a pre-built semantic knowledge base (JSON) containing text chunks and their embeddings.
2. Uses a SentenceTransformer model (same as the one used for chunking) to embed user queries.
3. Computes cosine similarity between the query embedding and chunk embeddings to find relevant chunks.
4. Returns and displays top matching chunks with similarity scores, cluster info, and content.
5. Provides an interactive command-line interface for users to enter queries, see results, get help, and view metadata.
6. Supports commands like 'help', 'config', and 'stats' for better user experience.

This tool enables semantic, content-based search over large text datasets prepared from PDFs or documents.
"""