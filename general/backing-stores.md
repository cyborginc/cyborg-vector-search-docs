# Backing Stores

Self-hosted versions of Cyborg Vector Search operate under a BYODB (Bring Your Own Database) model. In other words, Cyborg Vector Search sits between your application and your database backing store:

```
Diagram showing three-layered stack: application (sending data to be indexed & making queries), CVS, and backing stores
```

This allows Cyborg Vector Search to be a drop-in addition to existing stacks, without requiring the complexity of an additional storage layer.

## Supported Backing Stores

Cyborg Vector Search currently supports the following backing stores:

- MongoDB
- PostgreSQL
- Redis

In addition, for benchmarking purposes, the following backing store is available:

- Memory

Note that you can use either self-hosted or managed versions of the databases above (e.g., Amazon RDS).