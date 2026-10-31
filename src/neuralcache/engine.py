import numpy as np
import duckdb
from pydantic import BaseModel

class CacheEntry(BaseModel):
    query_id: str
    prompt: str
    response: str
    embedding: list[float]

class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("CREATE TABLE cache (query_id VARCHAR, prompt VARCHAR, response VARCHAR, embedding BLOB)")

    def put(self, entry: CacheEntry):
        vec_bytes = np.array(entry.embedding, dtype=np.float32).tobytes()
        self.conn.execute("INSERT INTO cache VALUES (?, ?, ?, ?)", [entry.query_id, entry.prompt, entry.response, vec_bytes])

    def query(self, query_embedding: list[float]) -> str | None:
        rows = self.conn.execute("SELECT query_id, prompt, response, embedding FROM cache").fetchall()
        if not rows: return None
        q_vec = np.array(query_embedding, dtype=np.float32)
        best_score = -1.0
        best_response = None
        for _, _, resp, vec_blob in rows:
            db_vec = np.frombuffer(vec_blob, dtype=np.float32)
            norm_prod = np.linalg.norm(q_vec) * np.linalg.norm(db_vec)
            score = float(np.dot(q_vec, db_vec) / norm_prod) if norm_prod > 0 else 0.0
            if score > best_score:
                best_score = score
                best_response = resp
        return best_response if best_score >= self.threshold else None
