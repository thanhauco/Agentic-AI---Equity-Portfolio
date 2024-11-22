"""
RAG Engine for AlphaAgents.

Implements Retrieval-Augmented Generation for SEC filings and financial reports.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

def _get_rag_components():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        return faiss, SentenceTransformer
    except ImportError:
        return None, None

class RAGEngine:
    """
    RAG Engine using FAISS for vector storage and Sentence-Transformers for embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents = []
        self._initialized = False
        
    def _initialize(self):
        if self._initialized:
            return
        faiss, SentenceTransformer = _get_rag_components()
        if SentenceTransformer:
            try:
                self.encoder = SentenceTransformer(self.model_name)
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to load RAG model: {e}")
        else:
            logger.warning("RAG dependencies (faiss, sentence-transformers) missing.")

    def add_documents(self, texts: List[str]):
        """Embed and index a list of document strings."""
        self._initialize()
        if not self.encoder:
            return
            
        try:
            embeddings = self.encoder.encode(texts)
            dim = embeddings.shape[1]
            
            import faiss
            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)
            
            self.index.add(np.array(embeddings).astype('float32'))
            self.documents.extend(texts)
            logger.info(f"Indexed {len(texts)} new document chunks")
        except Exception as e:
            logger.error(f"Indexing error: {e}")

    def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top k most relevant documents for a query."""
        self._initialize()
        if self.index is None or not self.encoder:
            return [{"text": "RAG engine not initialized or no documents indexed.", "score": 0}]
            
        try:
            query_vec = self.encoder.encode([query_text]).astype('float32')
            distances, indices = self.index.search(query_vec, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    results.append({
                        "text": self.documents[idx],
                        "score": float(distances[0][i])
                    })
            return results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return [{"text": f"Error during retrieval: {e}", "score": 0}]

    def ask(self, query: str, context_docs: Optional[List[str]] = None) -> str:
        """
        Build a contextual prompt for GenAI based on retrieved documents.
        This is intended to be used by an LLM-based Agent.
        """
        if context_docs is None:
            retrieved = self.query(query)
            context = "\n\n".join([r['text'] for r in retrieved])
        else:
            context = "\n\n".join(context_docs)
            
        prompt = f"""Use the following context from SEC filings to answer the user query.
If the answer is not in the context, say you don't know based on these documents.

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""
        return prompt
