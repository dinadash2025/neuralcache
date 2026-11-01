from neuralcache.engine import SemanticCache, CacheEntry

def test_semantic_cache_hit():
    cache = SemanticCache(threshold=0.80)
    cache.put(CacheEntry(query_id="1", prompt="What is Python?", response="A programming language.", embedding=[1.0, 0.0, 0.0]))
    
    # Exact/close vector query should hit
    hit = cache.query([0.95, 0.05, 0.0])
    assert hit == "A programming language."
