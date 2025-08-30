"""
Mock dependencies for testing when external services are not available
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock


class MockPinecone:
    """Mock Pinecone client for testing."""
    
    def __init__(self, api_key: str = "test-key", environment: str = "test"):
        self.api_key = api_key
        self.environment = environment
        self.indexes = {}
    
    def Index(self, name: str):
        """Return a mock index."""
        if name not in self.indexes:
            self.indexes[name] = MockPineconeIndex(name)
        return self.indexes[name]
    
    def list_indexes(self):
        """List available indexes."""
        return list(self.indexes.keys())


class MockPineconeIndex:
    """Mock Pinecone index for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.vectors = {}
    
    def upsert(self, vectors: List[Dict[str, Any]]):
        """Mock upsert operation."""
        for vector in vectors:
            self.vectors[vector["id"]] = vector
        return {"upserted_count": len(vectors)}
    
    def query(self, vector: List[float], top_k: int = 10, **kwargs):
        """Mock query operation."""
        # Return mock results
        return {
            "matches": [
                {
                    "id": f"mock_result_{i}",
                    "score": 0.9 - (i * 0.1),
                    "metadata": {"content": f"Mock content {i}"}
                }
                for i in range(min(top_k, 3))
            ]
        }
    
    def delete(self, ids: List[str]):
        """Mock delete operation."""
        deleted_count = 0
        for id in ids:
            if id in self.vectors:
                del self.vectors[id]
                deleted_count += 1
        return {"deleted_count": deleted_count}


def get_mock_pinecone():
    """Get a mock Pinecone client."""
    return MockPinecone()


def patch_pinecone_imports():
    """Patch Pinecone imports for testing."""
    import sys
    
    # Create mock pinecone module
    mock_pinecone = MagicMock()
    mock_pinecone.Pinecone = MockPinecone
    mock_pinecone.Index = MockPineconeIndex
    
    # Patch the import
    sys.modules['pinecone'] = mock_pinecone
    
    return mock_pinecone
