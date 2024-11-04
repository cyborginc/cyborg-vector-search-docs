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

The `CyborgVectorSearch` class provides an interface to initialize, train, and query encrypted vector indexes in Cyborg Vector Search. It allows creating different vector index types (such as `ivf`, `ivfpq`, `ivfflat`), with configurable distance metrics and device options. This Python binding for `CyborgVectorSearch` uses PyBind11, ensuring efficient integration with the C++ backend.

### Key Features

- **Vector Indexing**: Create, load, and manage vector indexes.
- **Encrypted Search**: Provides encrypted Approximate Nearest Neighbor (ANN) search capabilities.
- **Hardware Acceleration**: Option to configure CPU and GPU acceleration.
- **Flexible Data Input**: Accepts data in both `numpy` and `list` formats for embeddings and IDs.

## Constructor

```python
CyborgVectorSearch(index_location: dict,
                   config_location: dict,
                   items_location: Optional[dict] = None,
                   device_config: Optional[dict] = None)
```

Initializes a new `CyborgVectorSearch` instance.

**Parameters**:
| Parameter         | Type               | Description                                                         |
|-------------------|--------------------|---------------------------------------------------------------------|
| `index_location`  | [`LocationConfig (dict)`](#locationconfig)   | Configuration for index storage location. Use a dictionary with keys `location`, `table_name`, and `db_connection_string`. |
| `config_location` | [`LocationConfig (dict)`](#locationconfig)   | Configuration for index metadata storage, similar format as `index_location`. |
| `items_location`  | [`LocationConfig (dict)`](#locationconfig), optional | _(Optional)_ Configuration for item data storage. Defaults to `None`. |
| `device_config`   | [`DeviceConfig (dict)`](#deviceconfig), optional | _(Optional)_ Configuration for hardware acceleration. Defaults to `None`. |

**Example Usage**:

```python
index_location = {"location": "memory"}
config_location = {"location": "redis", "table_name": "index_metadata", "db_connection_string": "redis://localhost"}
device_config = {"cpu_threads": 4, "gpu_accelerate": True}

# Construct the CyborgVectorSearch object
search = CyborgVectorSearch(index_location=index_location,
                            config_location=config_location,
                            device_config=device_config)

# Proceed with further operations
```

## Create Index

```python
def create_index(self, index_name: str, index_key: bytes, index_config: dict)
```

Creates a new encrypted index based on the provided configuration.

**Parameters**:
| Parameter     | Type               | Description                                                                            |
|---------------|--------------------|----------------------------------------------------------------------------------------|
| `index_name`  | `str`              | Name of the index to create. Must be unique.                                           |
| `index_key`   | `bytes`            | 32-byte encryption key for the index.                                                  |
| `index_config`    | [`IndexConfig (dict)`](#indexconfig)         | Configuration dictionary specifying the index type (`ivf`, `ivfpq`, or `ivfflat`) and relevant parameters such as `dimension`, `n_lists`, `pq_dim`, and `pq_bits`. |

**Example Usage**:

```python
import secrets

search = CyborgVectorSearch(index_location=index_location, config_location=config_location)

index_name = "my_index"
index_key = secrets.token_bytes(32)  # Generate a secure 32-byte encryption key
index_config = {
    "type": "ivf",
    "dimension": 128,
    "n_lists": 1024
}

search.create_index(index_name=index_name, index_key=index_key, index_config=index_config)
```

## Load Index

```python
def load_index(self, index_name: str, index_key: bytes)
```

Connects to an existing encrypted index for further indexing or querying.

**Parameters**:
| Parameter    | Type    | Description                                                            |
|--------------|---------|------------------------------------------------------------------------|
| `index_name` | `str`   | Name of the index to load.                                             |
| `index_key`  | `bytes` | 32-byte encryption key used when the index was created.                |

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
def upsert(self, vectors: Union[Tuple[int, List[float]], List[Tuple[int, List[float]]]])
```

Adds or updates vector embeddings in the index.

### Primary Overload: Tuple Format
Accepts either a single tuple or a list of tuples, where each tuple represents a vector with its ID.

**Parameters**:
| Parameter | Type                              | Description                                                                                   |
|-----------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| `vectors` | `Tuple[int, List[float]]` or `List[Tuple[int, List[float]]]` | A tuple or list of tuples, where each tuple has: `(id: int, vector: List[float])`. |

- `id` (int): Unique integer identifier for the vector.
- `vector` (List[float]): Embedding vector as a list of floats.

**Example Usage**:

```python
# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# Single upsert
search.upsert((1, [0.1, 0.2, 0.3]))

# Batch upsert
search.upsert([
    (1, [0.1, 0.2, 0.3]),
    (2, [0.4, 0.5, 0.6])
])
```

### Secondary Overload: NumPy Array Format
```python
def upsert(self, vectors: np.ndarray, ids: List[int])
```
Accepts a NumPy array for embeddings and a list of integers for IDs, suitable for large batches.

**Parameters**:
| Parameter  | Type            | Description                                                                           |
|------------|-----------------|---------------------------------------------------------------------------------------|
| `vectors`  | `np.ndarray`    | 2D NumPy array of shape `(num_items, vector_dim)` representing vector embeddings.      |
| `ids`      | `List[int]`     | List of unique integer identifiers for each vector. Length must match `vectors`.      |

**Example Usage**:

```python
import numpy as np

# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# NumPy-based upsert
vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2 vectors of dimension 3
ids = [101, 102]  # Unique integer IDs for each vector

search.upsert(vectors=vectors, ids=ids)
```

**Exceptions**:
- Raises an exception if vector dimensions are incompatible with the index configuration.
- Raises an exception if the index has not been created or loaded.

## Train Index

```python
def train_index(self, training_config: Optional[dict] = None)
```

Trains the index based on the provided configuration. This step is necessary for efficient querying and should be called after enough vector embeddings have been upserted. If `train_index` is not called, queries will default to encrypted exhaustive search, which may be slower.

**Parameters**:
| Parameter          | Type     | Description                                                                                                  |
|--------------------|----------|--------------------------------------------------------------------------------------------------------------|
| `training_config` | [`TrainingConfig`](#trainingconfig), optional | _(Optional)_ Configuration for training parameters. This dictionary can include keys such as `batch_size`, `max_iters`, `tolerance`, and `max_memory`. |

- `batch_size` (int): Size of each batch for training. Defaults to `0` for auto-selection.
- `max_iters` (int): Maximum iterations for training. Defaults to `0` for auto-selection.
- `tolerance` (float): Convergence tolerance. Defaults to `1e-6`.
- `max_memory` (int): Maximum memory (MB) usage during training. Defaults to `0` (no limit).

**Example Usage**:

```python
# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# Define optional training configuration
training_config = {
    "batch_size": 128,
    "max_iters": 10,
    "tolerance": 1e-4,
    "max_memory": 1024
}

# Train the index with the configuration
search.train_index(training_config=training_config)

# Train with default settings (auto-selected configuration)
search.train_index()
```

**Exceptions**:
- Raises an exception if there are not enough vector embeddings in the index to support training.

## Query

```python
def query(self, query_vector: List[float], query_params: Optional[dict] = None) -> dict
def query(self, query_vectors: Union[np.ndarray, List[List[float]]], query_params: Optional[dict] = None) -> dict
```

Retrieves the nearest neighbors for one or more query vectors. This method provides two overloads:
- **Single query**: Accepts a single vector as a list of floats.
- **Batch query**: Accepts multiple query vectors as a 2D NumPy array or list of lists.

**Parameters**:
| Parameter       | Type                             | Description                                                                                  |
|-----------------|----------------------------------|----------------------------------------------------------------------------------------------|
| `query_vector`  | `List[float]`                    | A single query vector as a list of floats (for single query).                                |
| `query_vectors` | `np.ndarray` or `List[List[float]]` | A 2D NumPy array or list of lists, where each inner list represents a query vector (for batch query). |
| `query_params`  | [`QueryParams (dict)`](#queryparams), optional | _(Optional)_ Parameters for querying, including `top_k` and `n_probes`. |

- `top_k` (int): Number of nearest neighbors to return. Defaults to `100`.
- `n_probes` (int): Number of lists to probe during the query. Defaults to `1`.

**Returns**:
| Return Type   | Description                                                                     |
|---------------|---------------------------------------------------------------------------------|
| [`QueryResults (dict)`](#queryresults) | Dictionary containing `ids` and `distances` for nearest neighbors. |

- `ids`: List of lists of IDs for the nearest neighbors, corresponding to each query vector.
- `distances`: List of lists of distances for each nearest neighbor.

**Example Usage**:

_Single Query_:

```python
# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# Define a single query vector
query_vector = [0.1, 0.2, 0.3]  # Single query vector of dimension 3
query_params = {"top_k": 10, "n_probes": 5}

# Perform a single query
results = search.query(query_vector=query_vector, query_params=query_params)

# Access results
print("Nearest Neighbor IDs:", results["ids"])
print("Distances:", results["distances"])
```

_Batch Query_:

```python
import numpy as np

# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# Multiple query vectors
query_vectors = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])

# Query without additional parameters (using defaults)
results = search.query(query_vectors=query_vectors)

print("Nearest Neighbor IDs:", results["ids"])
print("Distances:", results["distances"])
```

## Delete Index

```python
def delete_index(self)
```

Deletes the current index and its associated data. This action is irreversible, so proceed with caution.

**Example Usage**:

```python
# Initial configuration
search = CyborgVectorSearch()
search.load_index()

# Delete the index
search.delete_index()
```

**Notes**:
- Deleting an index frees up resources associated with it.
- Ensure that the index is not in use elsewhere, as deleting it will make all operations dependent on this index invalid.

Here’s the **Getter Methods** section, covering each method to retrieve information about the index state.

## Getter Methods

### is_trained

```python
def is_trained(self) -> bool
```

Checks if the index has been trained. A trained index supports efficient approximate nearest neighbor search; an untrained index will use exhaustive search instead.

**Returns**:
| Return Type | Description                               |
|-------------|-------------------------------------------|
| `bool`      | `True` if the index has been trained; otherwise, `False`. |

**Example Usage**:

```python
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
| Return Type | Description                   |
|-------------|-------------------------------|
| `str`       | Name of the currently loaded or created index. |

**Example Usage**:

```python
print("Current index name:", search.index_name())
```

### index_type

```python
def index_type(self) -> str
```

Returns the type of the current index (e.g., `ivf`, `ivfpq`, `ivfflat`).

**Returns**:
| Return Type | Description                         |
|-------------|-------------------------------------|
| `str`       | Type of the current index.          |

**Example Usage**:

```python
print("Index type:", search.index_type())
```

### index_config

```python
def index_config(self) -> dict
```

Retrieves the configuration details of the current index.

**Returns**:
| Return Type | Description                       |
|-------------|-----------------------------------|
| `dict`      | Dictionary of index configuration parameters. |

**Example Usage**:

```python
config = search.index_config()
print("Index configuration:", config)
```

## Types

### LocationConfig

The `LocationConfig` dictionary specifies the storage location for the index, with options for in-memory storage, databases, or file-based storage.

**Structure**:
```python
{
    "location": str,                 # Specifies the storage type (e.g., "memory", "redis", "postgres", "mongodb").
    "table_name": Optional[str],     # (Optional) Name of the table in the database, if applicable.
    "db_connection_string": Optional[str]  # (Optional) Connection string for database access, if applicable.
}
```

The supported `location` options are:
- `"redis"`
- `"postgres"`
- `"mondogdb"`
- `"memory"` (for benchmarking and evaluation purposes)

For more info, you can read about supported backing stores [here](../general/backing-stores.md).

### DeviceConfig

The `DeviceConfig` dictionary specifies hardware options for running vector search operations, allowing control over CPU and GPU usage.

**Structure**:
```python
{
    "cpu_threads": int,               # Number of CPU threads for computations (0 = all cores).
    "gpu_accelerate": bool            # Enables GPU acceleration if available (True/False).
}
```

### DistanceMetric

`DistanceMetric` is a string representing the distance metric used for the index. Options include:
- `"cosine"`: Cosine similarity.
- `"euclidean"`: Euclidean distance.
- `"squared_euclidean"`: Squared Euclidean distance.

### IndexConfig

The `IndexConfig` dictionary defines the parameters for the type of index to be created. Each index type (e.g., `ivf`, `ivfflat`, `ivfpq`) has unique configuration options:

- **IVF** (Inverted File Index):
  ```python
  {
      "type": "ivf",
      "dimension": int,       # Number of dimensions in each vector.
      "n_lists": int,         # Number of inverted index lists.
      "metric": str           # Distance metric ("cosine", "euclidean", or "squared_euclidean").
  }
  ```

- **IVFFlat**:
  ```python
  {
      "type": "ivfflat",
      "dimension": int,
      "n_lists": int,
      "metric": str
  }
  ```

- **IVFPQ** (Inverted File with Product Quantization):
  ```python
  {
      "type": "ivfpq",
      "dimension": int,
      "n_lists": int,
      "pq_dim": int,          # Dimensionality after product quantization.
      "pq_bits": int,         # Number of bits per dimension (1-16).
      "metric": str
  }
  ```

### TrainingConfig

The `TrainingConfig` dictionary specifies training parameters to control convergence and memory usage.

**Structure**:
```python
{
    "batch_size": int,              # (Optional) Batch size for training. Defaults to 0 for auto-selection.
    "max_iters": int,               # (Optional) Maximum iterations. Defaults to 0 for auto-selection.
    "tolerance": float,             # (Optional) Convergence tolerance. Defaults to 1e-6.
    "max_memory": int               # (Optional) Maximum memory (MB) usage during training. Defaults to 0 (no limit).
}
```

### QueryParams

The `QueryParams` dictionary defines parameters for querying the index, controlling the number of results and probing behavior.

**Structure**:
```python
{
    "top_k": int,                   # (Optional) Number of nearest neighbors to return. Defaults to 100.
    "n_probes": int                 # (Optional) Number of lists to probe. Defaults to 1.
}
```

### QueryResults

The `QueryResults` dictionary contains the output from a `query` operation, including IDs and distances for each query’s nearest neighbors.

**Structure**:
```python
{
    "ids": List[List[int]],         # List of lists, where each inner list contains the IDs of nearest neighbors for a query vector.
    "distances": List[List[float]]  # List of lists, where each inner list contains distances of nearest neighbors for a query vector.
}
```