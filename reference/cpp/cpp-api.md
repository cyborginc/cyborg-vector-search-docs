# Cyborg Vector Search - C++ API

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
  - [Get Items](#get-items)
  - [Delete Index](#delete-index)
- [Getter Methods](#getter-methods)
- [Types](#types)

---

## Introduction

The C++ API for Cyborg Vector Search is split into two main classes within the `cyborg` namespace:

- **`Client`** – Handles configuration, index creation/loading, and listing available indexes.
- **`EncryptedIndex`** – Provides data operations on a specific encrypted index such as upserting vectors, training the index, querying, and retrieving stored items.

This API is also exposed via PyBind11 in the [Python API](../python/py-api.md).

---

## `Client`

The `cyborg::Client` class manages storage configurations and acts as a factory for creating or loading encrypted indexes.

### Constructor

```cpp
cyborg::Client(const LocationConfig& index_location,
               const LocationConfig& config_location,
               const LocationConfig& items_location,
               const int cpu_threads,
               const bool gpu_accelerate);
```
Initializes a new instance of `Client`.

**Parameters**:
| Parameter | Type | Description |
|-------------------|-----------------------|------------------------------------------------------|
| `index_location` | [`LocationConfig`](#locationconfig) | Configuration for index storage location. |
| `config_location` | [`LocationConfig`](#locationconfig) | Configuration for index metadata storage. |
| `items_location` | [`LocationConfig`](#locationconfig) | Configuration intended to be used in a future release. Pass in a LocationConfig with a Location of 'None'. |
| `cpu_threads` | `int` | Number of CPU threads to use (e.g., `0` to use all available cores).|
| `gpu_accelerate` | `bool` | Whether to enable GPU acceleration (requires CUDA).|

**Example Usage**:

```cpp
#include "client.hpp"

cyborg::LocationConfig index_location(Location::kMemory);
cyborg::LocationConfig config_location(Location::kRedis, "index_metadata", "redis://localhost");
cyborg::LocationConfig items_location(Location::kNone); // No item storage
int cpu_threads = 4;
bool use_gpu = true;

cyborg::Client client(index_loc, config_loc, items_loc, cpu_threads, use_gpu);
```

---

### Create Index

Creates and returns a new encrypted index based on the provided configuration.

```cpp
std::unique_ptr<cyborg::EncryptedIndex> CreateIndex(const std::string index_name,
                                                    const std::array<uint8_t, 32>& index_key,
                                                    const IndexConfig& index_config,
                                                    const std::optional<size_t>& max_cache_size);
```

**Parameters**:
| Parameter | Type | Description |
|----------------|-------------------------------|-----------------------------------------------------|
| `index_name` | `std::string` | Name of the index to create (must be unique). |
| `index_key` | `std::array<uint8_t, 32>` | 32-byte encryption key for the index, used to secure index data. |
| `index_config` | [`IndexConfig`](#indexconfig) | Configuration for the index type (e.g., IVFFlat, IVFPQ). |
| `max_cache_size` | `size_t` | _(Optional)_ Maximum size for the local cache (default is `0`). |

**Example Usage**:

```cpp
#include "client.hpp"
#include "encrypted_index.hpp"
#include <array>
#include <memory>
#include <optional>
#include <string>

// Create a secure 32-byte key (example: all zeros)
std::array<uint8_t, 32> index_key = {0};

// Example vector dimensionality & number of lists
const size_t vector_dim = 1024;
const size_t num_lists = 128;

// Create an index configuration (e.g., using an IVFFlat configuration)
IndexIVFFlat index_config(vector_dim, num_lists, DistanceMetric::euclidean);

auto index = client.CreateIndex("my_index", index_key, index_config);
```

---

### Load Index

Loads an existing encrypted index and returns an instance of `EncryptedIndex`.

```cpp
std::unique_ptr<cyborg::EncryptedIndex> LoadIndex(const std::string index_name,
                                                  const std::array<uint8_t, 32>& index_key,
                                                  const std::optional<size_t>& max_cache_size);
```

**Parameters**:
| Parameter | Type | Description |
|----------------|-------------------------------|-----------------------------------------------------|
| `index_name` | `std::string` | Name of the index to create (must be unique). |
| `index_key` | `std::array<uint8_t, 32>` | 32-byte encryption key for the index, used to secure index data. |
| `max_cache_size` | `size_t` | _(Optional)_ Maximum size for the local cache (default is `0`). |


**Example Usage**:

```cpp
auto index = client.LoadIndex("my_index", index_key, std::optional<size_t>{});
```

---

### List Indexes

Returns a list of all encrypted index names accessible via the client at the set `LocationConfig`.

```cpp
std::vector<std::string> ListIndexes();
```

**Example Usage**:

```cpp
auto indexes = client.ListIndexes();
for (const auto& name : indexes) {
    std::cout << name << std::endl;
}
```

---

## `EncryptedIndex`

The `cyborg::EncryptedIndex` class contains all data operations for a specific encrypted index.

### Upsert

Adds or updates vector embeddings in the index.

```cpp
void Upsert(Array2D<float>& vectors,
            const std::vector<uint64_t> ids,
            const std::vector<std::vector<uint8_t>> items = {});
```

**Parameters**:
| Parameter | Type | Description |
|---------------|-----------------------------|-----------------------------------------------------|
| `vectors` | [`Array2D<float>`](#array2d) | 2D container with vector embeddings to index. |
| `ids` | `std::vector<uint64_t>` | Unique identifiers for each vector. |
| `items` | `std::vector<std::vector<uint8_t>>` | _(Optional)_ Item contents in bytes. |

**Exceptions**:

- Throws if vector dimensions are incompatible with the index configuration.
- Throws if index was not created or loaded yet.
- Throws if there is a mismatch between the number of `vectors`, `ids` or `items`

**Example Usage**:

```cpp
#include "encrypted_index.hpp"

// Assume Array2D<float> is properly defined and populated.
cyborg::Array2D<float> embeddings{/*...*/};
std::vector<uint64_t> ids = {1, 2, 3};

// Upsert without additional item data
index->Upsert(embeddings, ids);

// Upsert with associated items
std::vector<std::vector<uint8_t>> items = {
    {'a', 'b', 'c'}, {'d', 'e', 'f'}, {'g', 'h', 'i'}
};
index->Upsert(embeddings, ids, items);
```

---

### Train

Builds the index using the specified training configuration. Required before efficient querying.
Prior to calling this, all queries will be conducted using encrypted exhaustive search.
After, they will be conducted using encrypted ANN search.

> [!IMPORTANT]
> This function is only present in the [embedded library](../../guides/0.overview/0.1.deployment-models.md) version of Cyborg Vector Search.
> In other versions (microservice, serverless), it is automatically called once enough vector embeddings have been indexed.


```cpp
void TrainIndex(const TrainingConfig& training_config = TrainingConfig());
```

**Parameters**:
| Parameter | Type | Description |
|----------|------------|--------------------|
| `training_config` | [`TrainingConfig`](#trainingconfig) | _(Optional)_ Training parameters (batch size, max iterations, etc.). |

**Exceptions**: Throws if there are not enough vector embeddings in the index for training (must be at least `2 * n_lists`).

**Example Usage**:

```cpp
cyborg::TrainingConfig config(128, 10, 1e-4, 1024);
index->TrainIndex(config);
```

> [!NOTE]
> There must be at least `2 * n_lists` vector embeddings in the index prior to to calling this function.

---

### Query

Retrieves the nearest neighbors for given query vectors.

```cpp
QueryResults Query(Array2D<float>& query_vectors,
                   const QueryParams& query_params = QueryParams());
```
Retrieves the nearest neighbors for given query vectors.

**Parameters**:
| Parameter | Type | Description |
|-------------------|--------------------|-----------------------|
| `query_vectors` | [`Array2D<float>`](#array2d) | Query vectors to search. |
| `query_params` | [`QueryParams`](#queryparams) | _(Optional)_ Parameters for querying, such as `top_k` and `n_lists`. |

**Returns**:
| Type | Description |
|-----|-----|
| [`QueryResults`](#queryresults) | Results `top_k` containing decrypted nearest neighbors IDs and distances. |

**Example Usage**:

```cpp
cyborg::Array2D<float> queries{/*...*/}; // Populate with one or more query vectors
cyborg::QueryParams params(10, 5);  // top_k = 10, n_probes = 5

QueryResults results = index->Query(queries, params);
std::cout << "ID: " << result.ids[j] << ", Distance: " << result.distances[j] << std::endl;
// Process query results...
```

> [!NOTE]
> If this function is called on an index where `TrainIndex()` has not been executed, the query will use encrypted exhaustive search.
> This may cause queries to be slower, especially when there are many vector embeddings in the index.

---

### Get Items

Retrieves and decrypts items associated with the specified IDs. (For a single item, pass a `std::vector` with one element.)

```cpp
std::vector<std::vector<uint8_t>> GetItems(std::vector<uint64_t> ids);
```

**Parameters**:
| Parameter | Type | Description |
|-------------------|--------------------|-----------------------|
| `ids` | `std::vector<uint64_t>` | IDs to retrieve. |

**Example Usage**:

```cpp
std::vector<uint64_t> item_ids = {1, 2, 3};
auto items = index->GetItems(item_ids);

for (const auto& item : items) {
    // Process each decrypted item (as a vector of bytes)
}
```

---

### Delete Index

Deletes the current index and all its associated data.  
**Warning**: This action is irreversible.

```cpp
void DeleteIndex();
```

Deletes the index and its associated data. Proceed with caution.

**Example Usage**:

```cpp
index->DeleteIndex();
```

---

## Getter Methods

The following methods provide information about the current state of the encrypted index:

### is_trained

```cpp
bool is_trained() const;
```

Returns whether the index has been trained.

---

### index_name

```cpp
std::string index_name() const;
```

Returns the name of the index.

---

### index_type

```cpp
IndexType index_type() const;
```

Returns the type of the index (for example, IVF, IVFPQ, or IVFFlat).

---

### index_config

```cpp
IndexConfig* index_config() const;
```

Returns a pointer to the index configuration.

**Example Usage**:

```cpp
if (index->is_trained()) {
    std::cout << "Index " << index->index_name() << " is trained." << std::endl;
}
```

---

## Types

### `Location`

The `Location` enum contains the supported index backing store locations for Cyborg Vector Search. These are:

```cpp
enum class Location {
    kRedis,      // In-memory storage via Redis
    kMemory,     // Temporary in-memory storage
    kPostgres,   // Relational database storage
    kNone        // Undefined storage type
};
```

---

### `LocationConfig`
_(Equivalent to the Python `DBConfig`)_  

`LocationConfig` defines the storage location for various index components.**Constructor**:

**Constructor**:

```cpp
LocationConfig(Location location,
                const std::optional<std::string>& table_name,
                const std::optional<std::string>& db_connection_string);
```

**Parameters**:
| Parameter | Type | Description |
|------------------------|-------------------------|-------------------------|
| `location` | [`Location`](#location) | Specifies the type of storage location. |
| `table_name` | `std::string` | _(Optional)_ Name of the table in the database, if applicable. |
| `db_connection_string` | `std::string` | _(Optional)_ Connection string for database access, if applicable. |


**Example Usage**:

```cpp
cyborg::LocationConfig index_loc(Location::kRedis, std::nullopt, "redis://localhost");
cyborg::LocationConfig config_loc(Location::kRedis, std::nullopt, "redis://localhost");
cyborg::LocationConfig items_loc(Location::kPostgres, "items", "host=localhost dbname=postgres");
```

---

### `DistanceMetric`

The `DistanceMetric` enum contains the supported distance metrics for Cyborg Vector Search. These are:

```cpp
enum class DistanceMetric {
    Cosine,
    Euclidean,
    SquaredEuclidean};
```

---


### `IndexConfig`

`IndexConfig` is an abstract base class for configuring index types. The three derived classes can be used to configure indexes:

#### `IndexIVF`

Ideal for large-scale datasets where fast retrieval is prioritized over high recall:

|  Speed  | Recall | Index Size |
| :-----: | :----: | :--------: |
| Fastest | Lowest |  Smallest  |

**Constructor**:

```cpp
IndexIVF(size_t dimension,
         size_t n_lists,
         DistanceMetric metric = DistanceMetric::Euclidean);
```

**Parameters**:
| Parameter | Type | Description |
|--------------|-------------------------|---------------------------------------|
| `dimension` | `size_t` | Dimensionality of vector embeddings. |
| `n_lists` | `size_t` | Number of inverted index lists to create in the index (recommended base-2 value).|
| `metric` | [`DistanceMetric`](#distancemetric) | _(Optional)_ Distance metric to use for index build and queries. |

For guidance on how to select the right `n_lists`, refer to the [index configuration tuning guide](../tuning-guides/index-configs.md).

#### `IndexIVFFlat`

Suitable for applications requiring high recall with less concern for memory usage:

| Speed | Recall  | Index Size |
| :---: | :-----: | :--------: |
| Fast  | Highest |  Biggest   |

**Constructor**:

```cpp
IndexIVFFlat(size_t dimension,
             size_t n_lists,
             DistanceMetric metric = DistanceMetric::Euclidean);
```

**Parameters**:
| Parameter | Type | Description |
|--------------|-------------------------------|-------------------------------------------------------------------------------|
| `dimension` | `size_t` | Dimensionality of vector embeddings. |
| `n_lists` | `size_t` | Number of inverted index lists to create in the index (recommended base-2 value).|
| `metric` | [`DistanceMetric`](#distancemetric) | _(Optional)_ Distance metric to use for index build and queries. |

For guidance on how to select the right `n_lists`, refer to the [index configuration tuning guide](../tuning-guides/index-configs.md).

#### `IndexIVFPQ`

Product Quantization compresses embeddings, making it suitable for balancing memory use and recall:

| Speed | Recall | Index Size |
| :---: | :----: | :--------: |
| Fast  |  High  |   Medium   |

**Constructor**:

```cpp
IndexIVFPQ(size_t dimension,
           size_t n_lists,
           size_t pq_dim,
           size_t pq_bits,
           DistanceMetric metric = DistanceMetric::Euclidean);
```

**Parameters**:
| Parameter | Type | Description |
|--------------|-------------------------------|-------------------------------------------------------------------------------|
| `dimension` | `size_t` | Dimensionality of vector embeddings. |
| `n_lists` | `size_t` | Number of inverted index lists to create in the index (recommended base-2 value).|
| `pq_dim` | `size_t` | Dimensionality of embeddings after quantization (<= dimension). |
| `pq_bits` | `size_t` | Number of bits per dimension for PQ embeddings (between 1 and 16). |
| `metric` | [`DistanceMetric`](#distancemetric) | _(Optional)_ Distance metric to use for index build and queries. |

For guidance on how to select the right `n_lists`, `pq_dim` and `pq_bits`, refer to the [index configuration tuning guide](../tuning-guides/index-configs.md).

---

### Array2D

`Array2D` class provides a 2D container for data, which can be initialized with a specific number of rows and columns, or from an existing vector.

**Constructors**:

```cpp
Array2D(size_t rows, size_t cols, const T& initial_value = T());
Array2D(std::vector<T>&& data, size_t cols);
Array2D(const std::vector<T>& data, size_t cols);
```

- **`Array2D(size_t rows, size_t cols, const T& initial_value = T())`**: Creates an empty 2D array with specified dimensions.
- **`Array2D(std::vector<T>&& data, size_t cols)`**: Initializes the 2D array from a 1D vector.
- **`Array2D(const std::vector<T>& data, size_t cols)`**: Initializes the 2D array from a 1D vector (copy).

**Access Methods**:

- **`operator()(size_t row, size_t col) const`**: Access an element at the specified row and column (read-only).
- **`operator()(size_t row, size_t col)`**: Access an element at the specified row and column (read-write).
- **`size_t rows() const`**: Returns the number of rows.
- **`size_t cols() const`**: Returns the number of columns.
- **`size_t size() const`**: Returns the total number of elements.

**Example Usage**:

```cpp
// Converting a vector to an array
std::vector<uint8_t> vec = {0, 1, 2, 3, 4, 5, 6, 7};
cyborg::Array2D<uint8_t> arr(vec, 2);
// arr is now a 2D array of 4 rows and 2 columns, with the contents from vec

// Creating a 2D array with 3 rows and 2 columns, initialized to zero
cyborg::Array2D<int> array(3, 2, 0);

// Access and modify elements
array(0, 0) = 1;
array(0, 1) = 2;

// Printing the array
for (size_t i = 0; i < array.rows(); ++i) {
    for (size_t j = 0; j < array.cols(); ++j) {
        std::cout << array(i, j) << " ";
    }
    std::cout << std::endl;
}
```

---

### `TrainingConfig`

The `TrainingConfig` struct defines parameters for training an index, allowing control over convergence and memory usage.

**Constructor**:

```cpp
TrainingConfig(size_t batch_size = 0,
                size_t max_iters = 0,
                double tolerance = 1e-6,
                size_t max_memory = 0);
```

**Parameters**:
| Parameter | Type | Description |
|-------------------|----------|---------------------------------------------------------------------------------------------|
| `batch_size` | `size_t` | _(Optional)_ Size of each batch for training. Defaults to `0`, which auto-selects the batch size. |
| `max_iters` | `size_t` | _(Optional)_ Maximum iterations for training. Defaults to `0`, which auto-selects iterations. |
| `tolerance` | `double` | _(Optional)_ Convergence tolerance for training. Defaults to `1e-6`. |
| `max_memory` | `size_t` | _(Optional)_ Maximum memory (MB) usage during training. Defaults to `0`, no limit. |

### `QueryParams`

The `QueryParams` struct defines parameters for querying the index, controlling the number of results and probing behavior.

**Constructor**:

```cpp
QueryParams(size_t top_k = 100,
            size_t n_probes = 1);
```

**Parameters**:
| Parameter | Type | Description |
|-------------------|----------|--------------------------------------------------------------------------|
| `top_k` | `size_t` | _(Optional)_ Number of nearest neighbors to return. Defaults to `100`. |
| `n_probes` | `size_t` | _(Optional)_ Number of lists to probe during query. Defaults to `1`. |
| `return_items` | `bool` | _(Optional)_ Whether to return item contents along with the IDs. Defaults to `false`. |
| `return_metadata` | `bool` | _(Optional)_ Whether to return metadata along with the IDs. Defaults to `false`. |
| `greedy` | `bool` | _(Optional)_ Whether to perform greedy search. Defaults to `false`. |

Higher n_probes values may improve recall but could slow down query time, so select a value based on desired recall and performance trade-offs. For guidance on how to select the right `n_probes`, refer to the [query parameter tuning guide](../tuning-guides/query-params.md).

---

### `QueryResults`

`QueryResults` class holds the results from a `Query` operation, including IDs and distances for the nearest neighbors of each query.

**Access Methods**:
| Method | Return Type | Description |
|--------------------------|------------------------------|------------------------------------------------------|
| `Result operator[](size_t query_idx)` | `Result` | Returns read-write access to IDs and distances for a specific query. |
| `const Array2D<uint64_t>& ids() const` | `const Array2D<uint64_t>&` | Get read-only access to all IDs. |
| `const Array2D<float>& distances() const` | `const Array2D<float>&` | Get read-only access to all distances. |
| `size_t num_queries() const` | `size_t` | Returns the number of queries. |
| `size_t top_k() const` | `size_t` | Returns the number of top-k items per query. |
| `bool empty() const` | `bool` | Checks if the results are empty. |

**Example Usage**:

```cpp
QueryResults results(num_queries, top_k);

// Access the top-k results for each query
for (size_t i = 0; i < num_queries; ++i) {
    auto result = results[i];
    for (size_t j = 0; j < result.num_results; ++j) {
        std::cout << "ID: " << result.ids[j] << ", Distance: " << result.distances[j] << std::endl;
    }
}

// Get the IDs and distances for all queries
auto all_ids = results.ids();
auto all_distances = results.distances();
```