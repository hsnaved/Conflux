#!/usr/bin/env python
"""Debug script to identify exact location of search error."""

import sys
import traceback

print("Python path:", sys.executable)
print("Python version:", sys.version)
print()

# Test 1: Import modules
print("[TEST 1] Importing vectorstore module...")
try:
    from services.vectorstore import get_client, search_similar, ensure_collection
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Get client
print("[TEST 2] Getting Qdrant client...")
try:
    client = get_client()
    print(f"  ✓ Client obtained: {type(client)}")
    print(f"  - Has search: {hasattr(client, 'search')}")
    print(f"  - Has query_points: {hasattr(client, 'query_points')}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Ensure collection
print("[TEST 3] Ensuring collection exists...")
try:
    ensure_collection()
    print("  ✓ Collection ensured")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create embedding
print("[TEST 4] Creating embedding...")
try:
    from services.embedding import embed_text
    vec = embed_text("test question")
    print(f"  ✓ Vector created, length: {len(vec)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Search
print("[TEST 5] Calling search_similar...")
try:
    results = search_similar(vec, limit=5)
    print(f"  ✓ Search succeeded, got {len(results)} results")
    for i, r in enumerate(results[:1], 1):
        payload_text = r.payload.get("text", "")[:100] if r.payload else ""
        print(f"    Result {i}: {payload_text}...")
except Exception as e:
    print(f"  ✗ Search failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
