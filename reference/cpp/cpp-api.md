# Cyborg Vector Search - C++ API

## Contents

- [Introduction](#introduction)
- [Constructor](#constructor)
- [Destructor](#destructor)
- [CreateIndex](#createindex)
- [LoadIndex](#loadindex)
- [Upsert](#upsert)
- [TrainIndex](#trainindex)
- [Query](#query)
- [DeleteIndex](#deleteindex)
- [Getter Methods](#getter-methods)
- [Types](#types)

---

## Introduction

The `CyborgVectorSearch` class is the core interface for creating and interacting with an encrypted index in Cyborg Vector Search. This class provides methods for initializing, training, and querying encrypted indexes. It supports various vector index types (e.g., IVF, IVFPQ, IVFFLAT) and can be configured with specific distance metrics and device options.

## Constructor

```cpp
cyborg::CyborgVectorSearch(const LocationConfig& index_location,
                   const LocationConfig& config_location,
                   const DeviceConfig& device_config = DeviceConfig());
```

Initializes a new instance of `CyborgVectorSearch`.

**Parameters**:
| Parameter | Type | Description |
|-------------------|-----------------------|------------------------------------------------------|
| `index_location` | [`LocationConfig`](#locationconfig) | Configuration for index storage location. |
| `config_location` | [`LocationConfig`](#locationconfig) | Configuration for index metadata storage. |
| `items_location` | [`LocationConfig`](#locationconfig) | Configuration intended to be used in a future release. Pass in a LocationConfig with a Location of 'None'. |
| `device_config` | [`DeviceConfig`](#deviceconfig) | _(Optional)_ Configuration for CPU and GPU acceleration.|

**Example Usage**:

```cpp
cyborg::LocationConfig index_location(Location::kMemory);
cyborg::LocationConfig config_location(Location::kRedis, "index_metadata", "redis://localhost");
cyborg::LocationConfig items_location(Location::kNone);
cyborg::DeviceConfig device_config(4, true); // Use 4 CPU threads and enable GPU acceleration

// Construct the CyborgVectorSearch object named as `search`
cyborg::CyborgVectorSearch search(index_location, config_location, items_location, device_config);

// Perform other operations...
```

## Destructor

```cpp
~cyborg::CyborgVectorSearch();
```

Destructs the `CyborgVectorSearch` object, releasing any allocated resources.

## CreateIndex

```cpp
void CreateIndex(const std::string index_name,
                 const std::array<uint8_t, 32>& index_key,
                 IndexConfig& index_config);
```

Creates a new index based on the provided configuration.

**Parameters**:
| Parameter | Type | Description |
|----------------|-------------------------------|-----------------------------------------------------|
| `index_name` | `std::string` | Name of the index to create (must be unique). |
| `index_key` | `std::array<uint8_t, 32>` | 32-byte encryption key for the index, used to secure index data. |
| `index_config` | [`IndexConfig`](#indexconfig) | Configuration for the index type (e.g., IVF, IVFPQ) |

**Example Usage**:

```cpp
cyborg::CyborgVectorSearch search(/*initial configurations*/);

const std::string index_name = "my_index";
std::array<uint8_t, 32> index_key = {/* 32-byte encryption key */};
IndexIVF index_config(128, 1024); // 128-dimensions, 1024 inverted lists

search.CreateIndex(index_name, index_key, index_config);
```

## LoadIndex

```cpp
void LoadIndex(const std::string index_name,
               const std::array<uint8_t, 32>& index_key);
```

Connects to an existing index for additional indexing or querying.

**Parameters**:
| Parameter | Type | Description |
|----------------|---------------------------|-----------------------------------------------------|
| `index_name` | `std::string` | Name of the index to load. |
| `index_key` | `std::array<uint8_t, 32>` | 32-byte encryption key for the index; must match the key used during [`CreateIndex()`](#createindex). |

**Example Usage**:

```cpp
cyborg::CyborgVectorSearch search(/*initial configurations*/);

std::string index_name = "my_index";
std::array<uint8_t, 32> index_key = {/* 32-byte encryption key */};

search.LoadIndex(index_name, index_key);
```

## Upsert

```cpp
void Upsert(Array2D<float>& vectors,
            const std::vector<uint64_t>& ids);
```

Ingests vector embeddings into the index.

**Parameters**:
| Parameter | Type | Description |
|---------------|-----------------------------|-----------------------------------------------------|
| `vectors` | [`Array2D<float>`](#array2d) | 2D container with vector embeddings to index. |
| `ids` | `std::vector<uint64_t>` | Unique identifiers for each vector. |

**Exceptions**:

- Throws if vector dimensions are incompatible with the index configuration.
- Throws if index was not created or loaded yet.
- Throws if there is a mismatch between the number of vectors and ids

**Example Usage**:

```cpp
cyborg::CyborgVectorSearch search(/*initial configurations*/);
search.LoadIndex(/*initial configurations*/);

Array2D<float> vectors = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}}; // 2 vectors of dimension 3
std::vector<uint64_t> ids = {101, 102};

search.Upsert(vectors, ids);
```

## TrainIndex

> [!IMPORTANT]
> This function is only present in the [embedded library](../../guides/0.overview/0.1.deployment-models.md) version of Cyborg Vector Search.
> In other versions (microservice, serverless), it is automatically called once enough vector embeddings have been indexed.

```cpp
void TrainIndex(const TrainingConfig& training_config = TrainingConfig());
```

Builds the index using the specified training configuration. Required before efficient querying.
Prior to calling this, all queries will be conducted using encrypted exhaustive search.
After, they will be conducted using encrypted ANN search.

**Parameters**:
| Parameter | Type | Description |
|----------|------------|--------------------|
| `training_config` | [`TrainingConfig`](#trainingconfig) | _(Optional)_ Training parameters (batch size, max iterations, etc.). |

**Exceptions**: Throws if there are not enough vector embeddings in the index for training (must be at least `2 * n_lists`).

**Example Usage**:

```cpp
cyborg::CyborgVectorSearch search(/*initial configurations*/);
search.LoadIndex(/*initial configurations*/);

cyborg::TrainingConfig training_config(128, 10, 1e-4, 1024); // batch size, max iters, tolerance, max memory (MB)
search.TrainIndex(training_config);
```

> [!NOTE]
> There must be at least `2 * n_lists` vector embeddings in the index prior to to calling this function.

## Query

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
cyborg::CyborgVectorSearch search(/*initial configurations*/);
search.LoadIndex(/*initial configurations*/);

Array2D<float> query_vectors = {{0.1f, 0.2f, 0.3f}};
QueryParams query_params(10, 5); // top-10 results, probe 5 lists

QueryResults results = search.Query(query_vectors, query_params);

std::cout << "ID: " << result.ids[j] << ", Distance: " << result.distances[j] << std::endl;
```

> [!NOTE]
> If this function is called on an index where `TrainIndex()` has not been executed, the query will use encrypted exhaustive search.
> This may cause queries to be slower, especially when there are many vector embeddings in the index.

## DeleteIndex

> [!WARNING]
> This action is irreversible and will erase all data associated with the index. Use with caution.

```cpp
void DeleteIndex();
```

Deletes the index and its associated data. Proceed with caution.

- **Example Usage**:

```cpp
cyborg::CyborgVectorSearch search(/*initial configurations*/);
search.LoadIndex(/*initial configurations*/);

search.DeleteIndex();
```

## Getter Methods

### is_trained

```cpp
bool is_trained() const;
```

Returns whether the index has been trained.

### index_name

```cpp
std::string index_name() const;
```

Returns the name of the current index.

### index_type

```cpp
IndexType index_type() const;
```

Returns the type of the current index.

### index_config

```cpp
IndexConfig* index_config() const;
```

Returns a pointer to the current index configuration.

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

### `LocationConfig`

The `LocationConfig` class configures storage locations for the index, including options for in-memory storage, databases, or file-based storage.

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

### `DeviceConfig`

The `DeviceConfig` class configures hardware usage for vector search operations, specifying CPU and GPU resources.

**Constructor**:

```cpp
explicit DeviceConfig(int cpu_threads = 0, bool gpu_accelerate = false);
```

**Parameters**:
| Parameter | Type | Description |
|-------------------|---------|---------------------------------|
| `cpu_threads` | `int` | Number of CPU threads to use for computations (defaults to `0` = all cores).|
| `gpu_accelerate` | `bool` | Indicates whether to use GPU acceleration (defaults to `false`). |

### `DistanceMetric`

The `DistanceMetric` enum contains the supported distance metrics for Cyborg Vector Search. These are:

```cpp
enum class DistanceMetric {
    Cosine,
    Euclidean,
    SquaredEuclidean};
```

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

Higher n_probes values may improve recall but could slow down query time, so select a value based on desired recall and performance trade-offs. For guidance on how to select the right `n_probes`, refer to the [query parameter tuning guide](../tuning-guides/query-params.md).

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
