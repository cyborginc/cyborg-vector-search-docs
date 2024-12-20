# Cyborg Vector Search - Versions & Release Roadmap

---

## v0.7

**Release Date**: 11/1/2024

**Deployment Models**: Embedded SDK (Python & C++)

**Features:**

- First public release of Cyborg Vector Search
- Python bindings available through PyBind
- Merged class (CyborgVectorSearch) for both Client and Index classes
- One set of keys for each index (no namespaces)
- GPU acceleration available through CUDA & cuVS


---

## v0.8

**Release Date**: December 2024

**Deployment Models**: Embedded SDK (Python & C++)

**Features:**

- Addition of encrypted item storage in indexes
	- Each index can handle item storage
	- `upsert()` calls can take item buffers, encrypt the items and store them in the backing store
	- `get_item()` and `get_items()` available for retrieval & decryption of encrypted items via `ids` (useful for applications like RAG which need the top matching items after query)
- Split API into two classes:
	- `Client` -> handles connections to backend DBs
	- `EncryptedIndex` -> returned from `create_index()` and/or `load_index()`
	- One client can handle many indexes (via multiple `EncryptedIndex` objects)
	- `list_indexes()`can list existing indexes available to the client
- Optimizations
	- Full-pipeline GPU acceleration
	- Optimized quantization & ranking logic
- Index cache with configurable size for faster queries
- Configurable logging utility