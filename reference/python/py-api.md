# Cyborg Vector Search - Python API

## Contents

- [Introduction](#introduction)
- [`Client`](#client)
  - [Constructor](#constructor)
  - [Create Index](#create-index)
  - [Load Index](#load-index)
  - [List Indexes](#list-indexes)
- [`EncryptedIndex`](#encryptedindex)
  - [Upsert](#upsert)
  - [Train](#train)
  - [Query](#query)
  - [Get Item](#get-item)
  - [Get Items](#get-items)
  - [Delete Index](#delete-index)
- [Getter Methods](#getter-methods)
- [Types](#types)
  - [DBConfig](#dbconfig)
  - [IndexTypes](#indextypes)
- [Optimized Overloads](#optimized-overloads)

---

## Introduction

The `CyborgVectorSearch` classes (`Client` and `EncryptedIndex`) provides an interface to initialize, train, and efficiently query encrypted vector indexes in Cyborg Vector Search. It allows creating different vector index types (such as `ivf`, `ivfpq`, `ivfflat`), with configurable distance metrics and device options. This Python binding for `CyborgVectorSearch` uses PyBind11, ensuring efficient integration with the C++ backend.

### Key Features

- **Vector Indexing**: Create, load, and manage vector indexes.
- **Encrypted Search**: Provides encrypted Approximate Nearest Neighbor (ANN) search capabilities.
- **Hardware Acceleration**: Option to configure CPU and GPU acceleration.
- **Flexible Data Input**: Accepts data in both `numpy` and `list` formats for embeddings and IDs.

---

## `Client`

`Client` is the main class exposed by `cyborg_vector_search_py`. It exposes the functionality necessary to create, load, list and delete indexes. Operations within encrypted indexes (such as `upsert` and `query`) are contained within the `EncryptedIndex` class returned by `create_index` and `load_index`.

## Constructor

```python
Client(index_location: dict,
       config_location: dict,
       item_location: dict,
       cpu_threads: int = 0,
       gpu_accelerate: bool = False)
```

Initializes a new Cyborg Vector Search `Client` instance.

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|--------------------|---------------------------------------------------------------------|---|
| `index_location` | [`DBConfig`](#dbconfig) | - | Configuration for index storage location. Use a dictionary with keys `location`, `table_name`, and `db_connection_string`. |
| `config_location` | [`DBConfig`](#dbconfig) | - | Configuration for index metadata storage. Uses the same dictionary structure as `index_location`. |
| `item_location` | [`DBConfig`](#dbconfig) | `NONE` | _(Optional)_ Configuration for encrypted item storage. Uses the same dictionary structure as `index_location`. |
| `cpu_threads` | `int` | `0` | _(Optional)_ Number of CPU threads to use for computations (defaults to `0` = all cores). |
| `gpu_accelerate` | `bool` | `False` | _(Optional)_ Indicates whether to use GPU acceleration (defaults to `False`). |

**Example Usage**:

```python
import cyborg_vector_search_py as cvs

index_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
config_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
item_location = cvs.DBConfig(location='postgres', table_name="items", connection_string="host=localhost dbname=postgres")

# Construct the Client object
client = cvs.Client(index_location=index_location,
                    config_location=config_location,
                    item_location=item_location,
                    cpu_threads=4,
                    gpu_accelerate=True)

# Proceed with further operations
```

## Create Index

```python
def create_index(self, index_name: str, index_key: bytes, index_config: dict)
```

Creates and returns a new encrypted index based on the provided configuration.

**Parameters**:
| Parameter | Type | Description |
|---------------|--------------------|----------------------------------------------------------------------------------------|
| `index_name` | `str` | Name of the index to create. Must be unique. |
| `index_key` | `bytes` | 32-byte encryption key for the index, used to secure the index data. |
| `index_config` | [`IndexConfig (dict)`](#indexconfig) | Configuration dictionary specifying the index type (`ivf`, `ivfpq`, or `ivfflat`) and relevant parameters such as `dimension`, `n_lists`, `pq_dim`, and `pq_bits`. |

**Example Usage**:

```python
import cyborg_vector_search_py as cvs
import secrets

index_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
config_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
item_location = cvs.DBConfig(location='postgres', table_name="items", connection_string="host=localhost dbname=postgres")

client = cvs.Client(index_location=index_location, config_location=config_location, item_location=item_location)

index_name = "my_index"
index_key = secrets.token_bytes(32)  # Generate a secure 32-byte encryption key
index_config = cvs.IndexIVF(dimension=128, n_lists=1024, metric="euclidean")

search.create_index(index_name=index_name, index_key=index_key, index_config=index_config)
```

## Load Index

```python
def load_index(self, index_name: str, index_key: bytes)
```

Connects to and returns an existing encrypted index for further indexing or querying.

**Parameters**:
| Parameter | Type | Description |
|--------------|---------|------------------------------------------------------------------------|
| `index_name` | `str` | Name of the index to load. |
| `index_key` | `bytes` | 32-byte encryption key; must match the key used during [`create_index()`](#create-index). |

**Example Usage**:

```python
import cyborg_vector_search_py as cvs

index_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
config_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")

client = cvs.Client(index_location=index_location, config_location=config_location)

index_name = "my_index"
index_key = my_index_key  # Use the same 32-byte encryption key used during index creation

# Load the existing index
index = client.load_index(index_name=index_name, index_key=index_key)
```

## List Indexes

```python
def list_indexes(self)
```

Returns a list of encrypted index names which are accessible to the client.

**Example Usage**:

```python
import cyborg_vector_search_py as cvs

index_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")
config_location = cvs.DBConfig(location='redis', connection_string="redis://localhost")

client = cvs.Client(index_location=index_location, config_location=config_location)

print(client.list_indexes())
# Example output:
# ["index_one", "index_two", "index_three"]
```

---

## `EncryptedIndex`

The `EncryptedIndex` class contains all data-related operations for an encrypted index.

## Upsert

```python
def upsert(self, vectors: List[Dict[str, Union[int, List[float], bytes, Dict[str, Union[str, int, bool, float]]]]])
```

Adds or updates vector embeddings in the index. Accepts a list of dictionaries, where each dictionary represents a vector with its ID.

**Parameters**:
| Parameter | Type | Description |
|-----------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| `vectors` | `List[Dict[str, Union[int, List[float], bytes, Dict[str, Union[str, int, bool, float]]]]]` | A list of dictionaries, where each dictionary has the format `{"id": int, "vector": List[float], "item": bytes, "metadata": Dict[]}`. |

- `id` (`int`): Unique integer identifier for the vector.
- `vector` (`List[float]`): Embedding vector as a list of floats.
- `item` (`bytes`): Item contents in bytes (_optional_)
- `metadata` (`dict`): Dictionary of key-value pairs associated with the vector (_optional_)

> [!TIP]
> For more info on metadata, see [Metadata Filtering](../guides/3.data-operations/3.3.metadata-filtering.md).
> Note that metadata filtering will not be available until `v0.9.0`

**Example Usage**:

```python
# Initial configuration already done...

# Load Index
index = client.load_index(index_name=index_name, index_key=index_key)

# Single upsert (wrapped in a list)
index.upsert([{"id": 1, "vector": [0.1, 0.2, 0.3]}])

# Upsert with metadata
index.upsert([
  {"id": 1,
   "vector": [0.1, 0.2, 0.3],
   "metadata": {"type": "dog", "temperament": "good boy"}
  }
])

# Batch upsert
index.upsert([
    {"id": 1, "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "vector": [0.4, 0.5, 0.6]}
])

# Upsert with items
index.upsert([
    {"id": 1, "vector": [0.1, 0.1, 0.1, 0.1], "item": b'item_contents_here...'},
    {"id": 2, "vector": [0.2, 0.2, 0.2, 0.2], "item": b'item_contents_here...'}
])
```

> [!TIP]
> You can pass the `id` and `vector` fields as tuples, if you wish to skip the dictionary. On big datasets, this can make a signficant difference in memory usage.

## Train

> [!IMPORTANT]
> This function is only present in the [embedded library](../guides/0.overview/0.1.deployment-models.md) version of Cyborg Vector Search.
> In other versions (microservice, serverless), it is automatically called once enough vector embeddings have been indexed.

```python
def train(self, batch_size: int = 0, max_iters: int = 0, tolerance: float = 1e-6, max_memory: int = 0)
```

Trains the index based on the provided configuration. This step is necessary for efficient querying and should be called after enough vector embeddings have been upserted. If `train` is not called, queries will default to encrypted exhaustive search, which may be slower.

**Parameters**:
| Parameter | Type | Default | Description |
|---------------|--------|---------|-------------------------------------------------------|
| `batch_size` | `int` | `0` | _(Optional)_ Size of each batch for training. `0` auto-selects the batch size. |
| `max_iters` | `int` | `0` | _(Optional)_ Maximum number of iterations for training. `0` auto-selects the iteration count. |
| `tolerance` | `float`| `1e-6` | _(Optional)_ Convergence tolerance for training. |
| `max_memory` | `int` | `0` | _(Optional)_ Maximum memory usage in MB during training. `0` imposes no limit. |

**Exceptions**:

- Raises an exception if there are not enough vector embeddings in the index to support training.

**Example Usage**:

```python
# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# Train the index with custom settings
index.train(batch_size=128, max_iters=10, tolerance=1e-4, max_memory=1024)

# Train with default settings (auto-selected configuration)
index.train()
```

> [!NOTE]
> There must be at least `2 * n_lists` vector embeddings in the index prior to calling this function.

## Query

### Single Query

```python
def query(self, 
          query_vector: List[float], 
          top_k: int = 100, 
          n_probes: int = 1, 
          filters: Optional[Dict[str, Any]] = None,
          return_distances: bool = True, 
          return_metadata: bool = False)
```

Retrieves the nearest neighbors for a given query vector.

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|----------------------------------|--------------|--------------------------------------------------------------------|
| `query_vector` | `List[float]` | - | A single query vector as a list of floats (for single query). |
| `top_k` | `int` | `100` | _(Optional)_ Number of nearest neighbors to return. |
| `n_probes` | `int` | `1` | _(Optional)_ Number of lists probed during the query; higher values may increase recall but can also reduce performance. |
| `filters` | `Dict[str, Any]` | `None` | _(Optional)_ A dictionary of filters to apply to vector metadata, limiting search scope to these vectors. |
| `return_distances`| `bool` | `True` | _(Optional)_ If `True`, each result includes `distance`. |
| `return_metadata` | `bool` | `False` | _(Optional)_ If `True`, each result includes `metadata` which contains the decrypted metadata fields, if available. | 

> [!NOTE]
> `filters` use a subset of the [MongoDB Query and Projection Operators](https://www.mongodb.com/docs/manual/reference/operator/query/). 
> For instance: `filters: { "$and": [ { "label": "cat" }, { "confidence": { "$gte": 0.9 } } ] }` means that only vectors where `label == "cat"` and `confidence >= 0.9` will be considered for encrypted vector search.
> Note that metadata filtering will not be available until `v0.9.0`

**Returns**:
| Return Type | Description |
|----------------------------|---------------------------------------------------------------------------------|
| `List[Dict[str, Union[int, float, Dict[]]]]` | List of results for the query vector. Each dictionary contains `id` and optionally `distance` if `return_distances` is `True`, and `metadata` if `return_metadata` is `True`. |

**Example Usage**:

_Single Query with Distances_:

```python
# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# Define a single query vector
query_vector = [0.1, 0.2, 0.3]  # Single query vector of dimension 3

# Perform a single query with distances
results = index.query(query_vector=query_vector, top_k=2, n_probes=5, return_distances=True)
# Example output:
# [{"id": 101, "distance": 0.05}, {"id": 102, "distance": 0.1}]
```

_Single Query without Distances_:

```python
results = index.query(query_vector=query_vector, top_k=10, n_probes=5, return_distances=False)
# Example output:
# [{"id": 101}, {"id": 102}]
```

_Single Query with Metadata_:

```python
results = search.query(query_vector=query_vector, top_k=10, n_probes=5, filters={"label": "dog"}, return_distances=False, return_metadata=True)
# Example output:
# [{"id": 101, "metadata": {"label": "dog", "temperament": "good boy"}}, 
#  {"id": 102, "metadata": {"label": "dog", "temperament": "hyper"})]]
```

### Batched Queries

```python
def query(self,
          query_vectors: Union[np.ndarray, List[List[float]]],
          top_k: int = 100,
          n_probes: int = 1,
          return_distances: bool = True)
```

Retrieves the nearest neighbors for one or more query vectors. 

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|----------------------------------|--------------|--------------------------------------------------------------------|
| `query_vectors` | `np.ndarray` or `List[List[float]]` | - | A 2D NumPy array or list of lists, where each inner list represents a query vector (for batch query). |
| `top_k` | `int` | `100` | _(Optional)_ Number of nearest neighbors to return. |
| `n_probes` | `int` | `1` | _(Optional)_ Number of lists probed during the query; higher values may increase recall but can also reduce performance. |
| `return_distances`| `bool` | `True` | _(Optional)_ If `True`, each result includes `(ID, distance)`. If `False`, only IDs are returned. |
| `return_metadata` | `bool` | `False` | _(Optional)_ If `True`, each result includes `metadata` which contains the decrypted metadata fields, if available. | 

**Returns**:
| Return Type | Description |
|----------------------------|---------------------------------------------------------------------------------|
| `List[List[Dict[str, Union[int, float, Dict[]]]]]` | List of results for each query vector. Each result is a list of `top_k`  dictionaries, each containing `id` and optionally `distance` if `return_distances` is `True`, and `metadata` if `return_metadata` is `True`. |

**Example Usage**:

_Batch Query with Distances_:

```python
import numpy as np

# Multiple query vectors
query_vectors = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])

# Query using default parameters and distances
results = index.query(query_vectors=query_vectors)
# Output example:
# [
#     [{"id": 101, "distance": 0.05}, {"id": 102, "distance": 0.1}],
#     [{"id": 201, "distance": 0.2}, {"id": 202, "distance": 0.3}]
# ]
```

## Get Item

```python
def get_item(self, id: int)
```

Retrieves, decrypts and returns an item from its item ID. If an item does not exist at that ID, it will return an empty `bytes` object.

**Parameters**:
| Parameter | Type  | Description |
|-------------------|----------------------------------|--------------|
| `id` | `int` | Item ID to retrieve & decrypt |

**Returns**:
| Return Type | Description |
|----------------------------|---------------------------------------------------------------------------------|
| `bytes` | Decrypted item bytes, or empty bytes object if no item was found at the ID provided. |

**Example Usage**:

```python
# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# Retrieve the item at ID '0'
item = index.get_item(0)

print(item)
# Example output:
# b'item contents here...'
```

## Get Items

```python
def get_items(self, ids: List[int])
```

Retrieves, decrypts and returns a list of items from their IDs. If an item does not exist at that ID, it will return an empty `bytes` object.

**Parameters**:
| Parameter | Type  | Description |
|-------------------|----------------------------------|--------------|
| `ids` | `List[int]` | Item IDs to retrieve & decrypt |

**Returns**:
| Return Type | Description |
|----------------------------|---------------------------------------------------------------------------------|
| `List[bytes]` | List of decrypted item bytes, or empty bytes object if no item was found at the ID provided. |

**Example Usage**:

```python
# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# Perform query
query_vector = [0.1, 0.2, 0.3]
results = index.query(query_vector=query_vector, top_k=10)

# Extract the item IDs from the query
result_ids = [res["id"] for res in results]

# Retrieve the items from the query results
items = index.get_items(result_ids)

print(items)
# Example output:
# [b'item #1 contents...', b'item #2 contents...', ...]
```


## Delete Index

> [!WARNING]
> This action is irreversible and will erase all data associated with the index. Use with caution.

```python
def delete_index(self)
```

Deletes the current index and its associated data. This action is irreversible, so proceed with caution.

**Example Usage**:

```python
# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# Delete the index
index.delete_index()
```

Here’s the **Getter Methods** section, covering each method to retrieve information about the index state.

## Getter Methods

The following methods retrieve information about the current state of the index, such as its name, type, and training status.

### is_trained

```python
def is_trained(self) -> bool
```

Returns `True` if the index has been trained, enabling efficient approximate nearest neighbor search. An untrained index will default to exhaustive search.

**Returns**:
| Return Type | Description |
|-------------|-------------------------------------------|
| `bool` | `True` if the index has been trained; otherwise, `False`. |

**Example Usage**:

```python
# Check if index is trained
if index.is_trained():
    print("The index is ready for efficient querying.")
else:
    print("The index is not trained; consider calling train().")
```

### index_name

```python
def index_name(self) -> str
```

Retrieves the name of the current index.

**Returns**:
| Return Type | Description |
|-------------|-------------------------------|
| `str` | Name of the currently loaded or created index. |

**Example Usage**:

```python
# Retrieve the index name
print("Current index name:", index.index_name())
```

### index_type

```python
def index_type(self) -> str
```

Returns the type of the current index (e.g., `ivf`, `ivfpq`, `ivfflat`).

**Returns**:
| Return Type | Description |
|-------------|-------------------------------------|
| `str` | Type of the current index. |

**Example Usage**:

```python
# Retrieve the type of index
print("Index type:", index.index_type())
```

### index_config

```python
def index_config(self) -> dict
```

Retrieves the configuration details of the current index.

**Returns**:
| Return Type | Description |
|-------------|-----------------------------------|
| `dict` | Dictionary of index configuration parameters. |

**Example Usage**:

```python
# Retrieve index config
print("Index configuration:", index.index_config())
```

---

## Types

### DBConfig

The `DBConfig` class specifies the storage location for the index, with options for in-memory storage, databases, or file-based storage.

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|----------------------------------|--------------|--------------------------------------------------------------------|
| `location` | `string` | - | DB location (`redis`, `postgres`, `memory`) |
| `table_name` | `string` | None | _(Optional)_ Table name (`postgres`-only) |
| `connection_string` | `string` | None | _(Optional)_ Connection string to access DB. |

The supported `location` options are:

- `"redis"`: Use for high-speed, in-memory storage (recommended for `index_location`).
- `"postgres"`: Use for reliable, SQL-based storage (recommended for `config_location`).
- `"memory"` Use for temporary in-memory storage (for benchmarking and evaluation purposes).

**Example Usage**:

```python
import cyborg_vector_search_py as cvs

index_location = cvs.DBConfig(location="redis", 
                          db_connection_string="redis://localhost")

config_location = cvs.DBConfig(location="postgres", 
                           table_name="config_table", db_connection_string="host=localhost dbname=postgres")
```

For more info, you can read about supported backing stores [here](../guides/0.overview/0.2.storage-locations.md).

### IndexTypes

The `IndexTypes` class defines the parameters for the type of index to be created. Each index type (e.g., `ivf`, `ivfflat`, `ivfpq`) has unique configuration options:

- **IVF** (Inverted File Index):

Ideal for large-scale datasets where fast retrieval is prioritized over high recall:

|  Speed  | Recall | Index Size |
| :-----: | :----: | :--------: |
| Fastest | Lowest |  Smallest  |

**Example Usage**:

  ```python
  import cyborg_vector_search_py as cvs

  index_config = cvs.IndexIVF(dimension=128, n_lists=1024, metric="euclidean")
  ```

- **IVFFlat**:

Suitable for applications requiring high recall with less concern for memory usage:

| Speed | Recall  | Index Size |
| :---: | :-----: | :--------: |
| Fast  | Highest |  Biggest   |

**Example Usage**:

  ```python
  import cyborg_vector_search_py as cvs

  index_config = cvs.IndexIVFFlat(dimension=128, n_lists=1024, metric="euclidean")
  ```

- **IVFPQ** (Inverted File with Product Quantization):

Product Quantization compresses embeddings, making it suitable for balancing memory use and recall:

| Speed | Recall | Index Size |
| :---: | :----: | :--------: |
| Fast  |  High  |   Medium   |

**Example Usage**:

  ```python
  import cyborg_vector_search_py as cvs

  index_config = cvs.IndexIVFPQ(dimension=128, n_lists=1024, pq_dim=64, pq_bits=8, metric="euclidean")
  ```

### DistanceMetric

`DistanceMetric` is a string representing the distance metric used for the index. Options include:

- `"cosine"`: Cosine similarity.
- `"euclidean"`: Euclidean distance.
- `"squared_euclidean"`: Squared Euclidean distance.

---

## Optimized Overloads

### `Upsert` Secondary Overload: NumPy Array Format

> [!TIP]
> This format is optimal for large batches due to its memory efficiency and compatibility with batch processing optimizations.

```python
def upsert(self, ids: np.ndarray, vectors: np.ndarray)
```

Accepts two NumPy arrays:

- A 2D array of floats for the vector embeddings.
- A 1D array of integers for the unique IDs.

This structure is suited for efficient handling of large batches, with type safety for IDs and embeddings.

**Parameters**:
| Parameter | Type | Description |
|------------|-----------------|--------------------------------------------------------------------------------------------------|
| `ids` | `np.ndarray` | 1D NumPy array of shape `(num_items,)` with `dtype=int`, containing unique integer identifiers for each vector. Length must match `vectors`. |
| `vectors` | `np.ndarray` | 2D NumPy array of shape `(num_items, vector_dim)` with `dtype=float`, representing vector embeddings. |


**Example Usage**:

```python
import numpy as np

# Load index
index = client.load_index(index_name=index_name, index_key=index_key)

# NumPy-based upsert with two arrays
vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)  # 2 vectors of dimension 3
ids = np.array([101, 102], dtype=int)  # Unique integer IDs for each vector

index.upsert(vectors=vectors, ids=ids)
```

**Exceptions**:

- Raises an exception if vector dimensions are incompatible with the index configuration.
- Raises an exception if the index has not been created or loaded.
