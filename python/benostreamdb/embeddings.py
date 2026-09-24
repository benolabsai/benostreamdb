import abc
from typing import List, Optional, Union, Dict, Any
import numpy as np

class EmbeddingFunction(abc.ABC):
    """Abstract base class for all embedding functions."""
    @abc.abstractmethod
    def __call__(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings into a numpy array of vectors."""
        pass

class HuggingFaceFunction(EmbeddingFunction):
    """
    Local embedding function using Sentence Transformers (supports all Hugging Face models).
    Examples: 'all-MiniLM-L6-v2', 'BAAI/bge-large-en-v1.5', 'Qwen/Qwen-7B-Chat' (if supported by ST)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", **kwargs):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name, device=device, **kwargs)

    def __call__(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Embedding function using the OpenAI API."""
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None, **kwargs):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        self.client = openai.OpenAI(api_key=api_key, **kwargs)
        self.model = model_name

    def __call__(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return np.array([data.embedding for data in response.data])

class AnthropicEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function using Anthropic/Claude (placeholder as Anthropic doesn't have a direct embedding API yet).
    Often used in conjunction with Voyage AI or similar.
    """
    def __init__(self, model_name: str = "voyage-2", api_key: Optional[str] = None, **kwargs):
        try:
            import voyageai
        except ImportError:
            raise ImportError("Anthropic often uses Voyage AI for embeddings. Please install: pip install voyageai")
        self.client = voyageai.Client(api_key=api_key)
        self.model = model_name

    def __call__(self, texts: List[str]) -> np.ndarray:
        result = self.client.embed(texts, model=self.model)
        return np.array(result.embeddings)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function using Google's Gemini API."""
    def __init__(self, model_name: str = "models/embedding-001", api_key: Optional[str] = None, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = model_name
        self.kwargs = kwargs

    def __call__(self, texts: List[str]) -> np.ndarray:
        import google.generativeai as genai
        result = genai.embed_content(model=self.model, content=texts, **self.kwargs)
        return np.array(result['embedding'])

class EmbeddingRegistry:
    """Registry to manage and retrieve embedding functions."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingRegistry, cls).__new__(cls)
            cls._instance.functions = {}
        return cls._instance
    
    def register(self, name: str, func: EmbeddingFunction):
        """Register a new embedding function."""
        self.functions[name] = func
        
    def get(self, name: str) -> Optional[EmbeddingFunction]:
        """Retrieve a registered embedding function."""
        return self.functions.get(name)

# Global registry instance
registry = EmbeddingRegistry()

def get_registry():
    """Access the global embedding registry."""
    return registry
