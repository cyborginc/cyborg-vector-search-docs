# Cyborg Vector Search - Getting Started in Python

## Installation

1. Ensure Python is installed (currently `3.11` and `3.12` are supported)

2. If your version does not satisfy the requirement, create a compatible environment:

    ```bash
    conda create -n cyborg-env python=3.11
    ```

3. Install Cyborg Vector Search:

    ```bash
    pip install cyborg_vector_search_py -i https://dl.cloudsmith.io/<token>/cyborg/cyborg-vector-search/python/simple/
    ```

> [!IMPORTANT]  
> You will need to replace `<token>` with the token provided by the Cyborg team for your use. Without it, you cannot install the `cyborg_vector_search_py` module.

## Example Notebooks

To get started quickly, we invite you to try running one of our example notebooks:

- [Example with Memory Backing Store >](example-notebooks/memory-example.ipynb)
- [Example with Redis Backing Store >](example-notebooks/redis-example.ipynb)

Alternatively, follow the steps below and review our [Python API guide](py-api.md) to learn more.

## Cyborg Vector Search Usage

### 1. Importing Cyborg Vector Search

Import CVS in your Python file:
```python
import cyborg_vector_search_py as cvs
```

### 2. Initializing CVS
Initialize the CVS client with the configuration:
```python
index_location = cvs.LocationConfig(cvs.Location.MEMORY)
config_location = cvs.LocationConfig(cvs.Location.MEMORY)

client = cvs.CyborgVectorSearch(
    index_location=index_location,
    config_location=config_location,
    cpu_threads=10,
    gpu_accelerate=False
)

```

### 2. Creating an Index

Configure the index location and setup the index configuration:

```python
# Select index location
index_location = cvs.LocationConfig(cvs.Location.MEMORY)
config_location = cvs.LocationConfig(cvs.Location.MEMORY)

# Set index parameters
dimension = 128  # Adjust according to your dataset
n_lists = 4096
metric = "euclidean"
index_config = cvs.IndexIVFFlat(dimension, n_lists, metric, False)
index_name = "test_index"
index_key = bytes([0] * 32) # Set your private key here

# Create the index
client.create_index(index_name, index_key, index_config)
```

### 4. Upserting Vectors
Insert vectors into the index:

```python
client.upsert(vectors, ids)
```

### 5. Querying the Index

Query with untrained or trained vectors, adjusting parameters as needed:

```python
top_k = 100
n_probes = 10
results = client.query(test_vectors, top_k, n_probes)
```

## Further Reading

For more details, please reivew our [example notebooks](example-notebooks/) or [Python API guide](py-api.md).