# Cyborg Vector Search - Python API

## Contents

- [Introduction](#introduction)
- [Constructor](#constructor)
- [Create Index](#create-index)
- [Load Index](#load-index)
- [Upsert](#upsert)
- [Train Index](#train-index)
- [Query](#query)
- [Delete Index](#delete-index)
- [Getter Methods](#getter-methods)
- [Types](#types)

---

## Introduction

The `CyborgVectorSearch` class provides an interface to initialize, train, and efficiently query encrypted vector indexes in Cyborg Vector Search. It allows creating different vector index types (such as `ivf`, `ivfpq`, `ivfflat`), with configurable distance metrics and device options. This Python binding for `CyborgVectorSearch` uses PyBind11, ensuring efficient integration with the C++ backend.

### Key Features

- **Vector Indexing**: Create, load, and manage vector indexes.
- **Encrypted Search**: Provides encrypted Approximate Nearest Neighbor (ANN) search capabilities.
- **Hardware Acceleration**: Option to configure CPU and GPU acceleration.
- **Flexible Data Input**: Accepts data in both `numpy` and `list` formats for embeddings and IDs.

## Constructor

```python
CyborgVectorSearch(index_location: dict,
                   config_location: dict,
                   cpu_threads: int = 0,
                   gpu_accelerate: bool = False)
```

Initializes a new `CyborgVectorSearch` instance.

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|--------------------|---------------------------------------------------------------------|---|
| `index_location` | [`LocationConfig (dict)`](#locationconfig) | - | Configuration for index storage location. Use a dictionary with keys `location`, `table_name`, and `db_connection_string`. |
| `config_location` | [`LocationConfig (dict)`](#locationconfig) | - | Configuration for index metadata storage. Uses the same dictionary structure as `index_location`. |
| `cpu_threads` | `int` | `0` | _(Optional)_ Number of CPU threads to use for computations (defaults to `0` = all cores). |
| `gpu_accelerate` | `bool` | `False` | _(Optional)_ Indicates whether to use GPU acceleration (defaults to `False`). |

**Example Usage**:

```python
index_location = LocationConfig(Location.MEMORY)
config_location = LocationConfig(Location.REDIS, table_name="index_metadata", db_connection_string="redis://localhost")

# Construct the CyborgVectorSearch object
search = CyborgVectorSearch(index_location=index_location, config_location=config_location, cpu_threads=4, gpu_accelerate=True)

# Proceed with further operations
```

## Create Index

```python
def create_index(self, index_name: str, index_key: bytes, index_config: dict)
```

Creates a new encrypted index based on the provided configuration.

**Parameters**:
| Parameter | Type | Description |
|---------------|--------------------|----------------------------------------------------------------------------------------|
| `index_name` | `str` | Name of the index to create. Must be unique. |
| `index_key` | `bytes` | 32-byte encryption key for the index, used to secure the index data. |
| `index_config` | [`IndexConfig (dict)`](#indexconfig) | Configuration dictionary specifying the index type (`ivf`, `ivfpq`, or `ivfflat`) and relevant parameters such as `dimension`, `n_lists`, `pq_dim`, and `pq_bits`. |

**Example Usage**:

```python
import secrets

search = CyborgVectorSearch(index_location=index_location, config_location=config_location)

index_name = "my_index"
index_key = secrets.token_bytes(32)  # Generate a secure 32-byte encryption key
index_config = IndexIVF(dimension=128, n_lists=1024, metric="euclidean", store_items=False)

search.create_index(index_name=index_name, index_key=index_key, index_config=index_config)
```

## Load Index

```python
def load_index(self, index_name: str, index_key: bytes)
```

Connects to an existing encrypted index for further indexing or querying.

**Parameters**:
| Parameter | Type | Description |
|--------------|---------|------------------------------------------------------------------------|
| `index_name` | `str` | Name of the index to load. |
| `index_key` | `bytes` | 32-byte encryption key; must match the key used during [`create_index()`](#create-index). |

**Example Usage**:

```python
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)

index_name = "my_index"
index_key = my_index_key  # Use the same 32-byte encryption key used during index creation

# Load the existing index
search.load_index(index_name=index_name, index_key=index_key)
```

## Upsert

```python
def upsert(self, vectors: List[Tuple[int, List[float]]])
```

Adds or updates vector embeddings in the index. Accepts a list of tuples, where each tuple represents a vector with its ID.

**Parameters**:
| Parameter | Type | Description |
|-----------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| `vectors` | `List[Tuple[int, List[float]]]` | A list of tuples, where each tuple has the format `(id: int, vector: List[float])`. |

- `id` (int): Unique integer identifier for the vector.
- `vector` (List[float]): Embedding vector as a list of floats.

**Example Usage**:

```python
# Initial configuration
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)
search.load_index(index_name=index_name, index_key=index_key)

# Single upsert (wrapped in a list)
search.upsert([(1, [0.1, 0.2, 0.3])])

# Batch upsert
search.upsert([
    (1, [0.1, 0.2, 0.3]),
    (2, [0.4, 0.5, 0.6])
])
```

### Secondary Overload: NumPy Array Format

> [!TIP]
> This format is optimal for large batches due to its memory efficiency and compatibility with batch processing optimizations.

```python
def upsert(self, vectors: np.ndarray, ids: np.ndarray)
```

Accepts two NumPy arrays:

- A 2D array of floats for the vector embeddings.
- A 1D array of integers for the unique IDs.

This structure is suited for efficient handling of large batches, with type safety for IDs and embeddings.

**Parameters**:
| Parameter | Type | Description |
|------------|-----------------|--------------------------------------------------------------------------------------------------|
| `vectors` | `np.ndarray` | 2D NumPy array of shape `(num_items, vector_dim)` with `dtype=float`, representing vector embeddings. |
| `ids` | `np.ndarray` | 1D NumPy array of shape `(num_items,)` with `dtype=int`, containing unique integer identifiers for each vector. Length must match `vectors`. |

**Example Usage**:

```python
import numpy as np

# Initial configuration
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)
search.load_index(index_name=index_name, index_key=index_key)

# NumPy-based upsert with two arrays
vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)  # 2 vectors of dimension 3
ids = np.array([101, 102], dtype=int)  # Unique integer IDs for each vector

search.upsert(vectors=vectors, ids=ids)
```

**Exceptions**:

- Raises an exception if vector dimensions are incompatible with the index configuration.
- Raises an exception if the index has not been created or loaded.

## Train Index

> [!IMPORTANT]
> This function is only present in the [embedded library](../general/deployment-models.md) version of Cyborg Vector Search.
> In other versions (microservice, serverless), it is automatically called once enough vector embeddings have been indexed.

```python
def train_index(self, batch_size: int = 0, max_iters: int = 0, tolerance: float = 1e-6, max_memory: int = 0)
```

Trains the index based on the provided configuration. This step is necessary for efficient querying and should be called after enough vector embeddings have been upserted. If `train_index` is not called, queries will default to encrypted exhaustive search, which may be slower.

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
# Initial configuration
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)
search.load_index(index_name=index_name, index_key=index_key)

# Train the index with custom settings
search.train_index(batch_size=128, max_iters=10, tolerance=1e-4, max_memory=1024)

# Train with default settings (auto-selected configuration)
search.train_index()
```

> [!NOTE]
> There must be at least `2 * n_lists` vector embeddings in the index prior to calling this function.

## Query

```python
def query(self, query_vectors: List[float], top_k: int = 100, n_probes: int = 1, return_distances: bool = True) -> List[List[Union[int, Tuple[int, float]]]]
def query(self, query_vectors: Union[np.ndarray, List[List[float]]], top_k: int = 100, n_probes: int = 1, return_distances: bool = True) -> List[List[Union[int, Tuple[int, float]]]]
```

Retrieves the nearest neighbors for one or more query vectors. This method provides two overloads:

- **Single query**: Accepts a single vector as a list of floats.
- **Batch query**: Accepts multiple query vectors as a 2D NumPy array or list of lists.

**Parameters**:
| Parameter | Type | Default | Description |
|-------------------|----------------------------------|--------------|--------------------------------------------------------------------|
| `query_vectors` | `List[float]` | - | A single query vector as a list of floats (for single query). |
| `query_vectors` | `np.ndarray` or `List[List[float]]` | - | A 2D NumPy array or list of lists, where each inner list represents a query vector (for batch query). |
| `top_k` | `int` | `100` | _(Optional)_ Number of nearest neighbors to return. |
| `n_probes` | `int` | `1` | _(Optional)_ Number of lists probed during the query; higher values may increase recall but can also reduce performance. |
| `return_distances`| `bool` | `True` | _(Optional)_ If `True`, each result includes `(ID, distance)`. If `False`, only IDs are returned. |

**Returns**:
| Return Type | Description |
|----------------------------|---------------------------------------------------------------------------------|
| `List[List[Tuple[int, float]]]` | List of results for each query vector if `return_distances` is `True`. Each result is a list of `top_k` `(ID, distance)` tuples. |
| `List[List[int]]` | List of results for each query vector if `return_distances` is `False`. Each result is a list of `top_k` IDs. |

**Example Usage**:

_Single Query with Distances_:

```python
# Initial configuration
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)
search.load_index(index_name=index_name, index_key=index_key)

# Define a single query vector
query_vectors = [0.1, 0.2, 0.3]  # Single query vector of dimension 3

# Perform a single query with distances
results = search.query(query_vector=query_vectors, top_k=2, n_probes=5, return_distances=True)
# Output example:
# [[(101, 0.05), (102, 0.1)]]
```

_Single Query without Distances_:

```python
results = search.query(query_vector=query_vectors, top_k=10, n_probes=5, return_distances=False)
# Example output:
# [[101, 102]]
```

_Batch Query with Distances_:

```python
import numpy as np

# Multiple query vectors
query_vectors = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])

# Query using default parameters and distances
results = search.query(query_vectors=query_vectors)
# Output example:
# [[(101, 0.05), (102, 0.1)], [(201, 0.2), (202, 0.3)]]
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
# Initial configuration
search = CyborgVectorSearch(index_location=index_location, config_location=config_location)
search.load_index(index_name=index_name, index_key=index_key)

# Delete the index
search.delete_index()
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
if search.is_trained():
    print("The index is ready for efficient querying.")
else:
    print("The index is not trained; consider calling train_index().")
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
print("Current index name:", search.index_name())
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
print("Index type:", search.index_type())
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
print("Index configuration:", search.index_config())
```

## Types

### LocationConfig

The `LocationConfig` dictionary specifies the storage location for the index, with options for in-memory storage, databases, or file-based storage.

**Structure**:

```python
{
    "location": str,                 # Specifies the storage type (e.g., "memory", "redis", "postgres").
    "table_name": Optional[str],     # (Optional) Name of the table in the database, if applicable.
    "db_connection_string": Optional[str]  # (Optional) Connection string for database access, if applicable.
}
```

The supported `location` options are:

- `"REDIS"`: Use for high-speed, in-memory storage (recommended for `index_location`).
- `"POSTGRES"`: Use for reliable, SQL-based storage (recommended for `config_location`).
- `"MEMORY"` Use for temporary in-memory storage (for benchmarking and evaluation purposes).

For more info, you can read about supported backing stores [here](../general/backing-stores.md).

### DistanceMetric

`DistanceMetric` is a string representing the distance metric used for the index. Options include:

- `"cosine"`: Cosine similarity.
- `"euclidean"`: Euclidean distance.
- `"squared_euclidean"`: Squared Euclidean distance.

### IndexConfig

The `IndexConfig` dictionary defines the parameters for the type of index to be created. Each index type (e.g., `ivf`, `ivfflat`, `ivfpq`) has unique configuration options:

- **IVF** (Inverted File Index):

Ideal for large-scale datasets where fast retrieval is prioritized over high recall:

|  Speed  | Recall | Index Size |
| :-----: | :----: | :--------: |
| Fastest | Lowest |  Smallest  |

**Example Usage**:

  ```python
  index_config = IndexIVF(dimension=128, n_lists=1024, metric="euclidean", store_items=False)
  ```

- **IVFFlat**:

Suitable for applications requiring high recall with less concern for memory usage:

| Speed | Recall  | Index Size |
| :---: | :-----: | :--------: |
| Fast  | Highest |  Biggest   |

**Example Usage**:

  ```python
  index_config = IndexIVFFlat(dimension=128, n_lists=1024, metric="euclidean", store_items=False)
  ```

- **IVFPQ** (Inverted File with Product Quantization):

Product Quantization compresses embeddings, making it suitable for balancing memory use and recall:

| Speed | Recall | Index Size |
| :---: | :----: | :--------: |
| Fast  |  High  |   Medium   |

**Example Usage**:

  ```python
  index_config = IndexIVFPQ(dimension=128, n_lists=1024, pq_dim=64, pq_bits=8, metric="euclidean", store_items=False)
  ```