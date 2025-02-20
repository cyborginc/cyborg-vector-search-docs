# CyborgDB Docs

> [!IMPORTANT]  
> These docs are still under construction as of `v0.9.0` and may change until `v1.0.0` is released.

**CyborgDB** is the first Confidential Vector DB:

- Introduces a novel architecture to keep confidential inference data secure through zero-trust design.
- Keeps vector embeddings end-to-end encrypted throughout their lifecycle (including at search time).
- Exposes a familiar API, making it easy to migrate from traditional Vector DBs.
- Provides high-performance indexing and retrieval which can be [GPU-accelerated with CUDA](https://developer.nvidia.com/blog/bringing-confidentiality-to-vector-search-with-cyborg-and-nvidia-cuvs/).
- Works with many backend DBs such as Postgres and Redis, integrating easily into your existing infrastructure.

[Learn more about CyborgDB >](guides/0.overview/0.0.overview.md)

## Getting Started

To get started with CyborgDB, follow these steps:

1. Request access to our private `PyPI` repo by emailing us at [earlyaccess@cyborg.co](mailto:earlyaccess@cyborg.co?subject=Early%20Access%20Request%20-%20Cyborg%20Vector%20Search).

2. Download the release artifacts from the private repo (`pip install cyborgdb -i <private_repo>`).

3. Get started with our [Quickstart Guide](guides/1.getting-started/1.0.quickstart.md) or review the API Docs for [Python](reference/python/) or [C++](reference/cpp/) to start using CyborgDB!

## Further Reading

- [Intro to CyborgDB >](guides/0.overview/0.0.overview.md)
- [Guides >](guides/)
- [Python API >](reference/python/py-api.md)
- [Python Examples >](examples/python/)
- [C++ API >](reference/cpp/cpp-api.md)
- [C++ Examples (coming soon) >](examples/cpp/)
- [Performance Benchmarks (coming soon) >](benchmarks/)
- [Releases (coming soon) >](releases/version-roadmap.md)

## License

All contents of this repo are the exclusive property of [Cyborg Inc.](https://www.cyborg.co)
