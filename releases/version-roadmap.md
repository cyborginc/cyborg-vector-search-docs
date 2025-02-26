# CyborgDB - Versions & Release Roadmap

---

## v0.7

**Release Date**: 11/1/2024

**Deployment Models**: Embedded SDK (Python & C++)

**Features**:

- First public release of CyborgDB
- Python bindings available through PyBind
- Merged class (CyborgDB) for both Client and Index classes
- One set of keys for each index (no namespaces)
- GPU acceleration available through CUDA & cuVS

---

## v0.8

**Release Date**: 12/19/2024

**Deployment Models**: Embedded SDK (Python & C++)

**Features**:

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

---

## v0.9

**Release Date**: February 2025

**Deployment Models**: Embedded SDK (Python & C++)

**Features**:

- Addition of metadata fields & query filtering
  - Each item can have metadata fields added in `upsert()`
  - `query()` calls can take a `filter` dictionary of metadata fields, values and operators
- Item IDs are now `string`-type instead of `int`
- Addition of item deletion functionality
  - `delete()` can take one or more item IDs and delete their associated contents/fields
  - `upsert()` now properly *update* items on ID conflicts
- Addition of managed embedding generation
  - `embedding_model` can be defined on `create_index()` with a `sentence-transformers` model name
  - If defined, `upsert()` and `query()` can automatically generate embeddings instead of providing `vector`
  - Depending on model, both `text` and `images` are supported