# Cyborg Vector Search

## Contents

- [Introduction](#introduction)
- [Why Confidential Vector Search?](#why-confidential-vector-search)
- [How It Works](#how-it-works)
- [Further Reading](#further-reading)

## Introduction

**Cyborg Vector Search** is the first Confidential Vector DB:

- Introduces a novel architecture to keep confidential inference data secure through zero-trust design.
- Keeps vector embeddings end-to-end encrypted throughout their lifecycle (including at search time).
- Exposes a familiar API, making it easy to migrate from traditional Vector DBs.
- Provides high-performance indexing and retrieval which can be GPU-accelerated with CUDA.

## Why Confidential Vector Search?

**Why _Confidential_?**

According to KPMG, 63% of enterprises say that confidentiality and data privacy are their top risk to AI adoption. In regulated sectors, this figure increases to 76%. Yet, when it comes to solutions which address these concerns, the market has yet to answer. This leaves a critical mass of companies unserved and unable to adopt AI in their workflows. They need **Confidential AI**.

**Why _Confidential Vector Search_?**

Vector Search is at the heart of the most popular AI applications - RAG, RecSys, Semantic Search, IR, etc. - and the market is overcrowded with Vector DBs. All of these share a key commonality: they need to store their indexed contents (vector embeddings) in plaintext to enable Approximate Nearest-Neighbor (ANN) search. This creates a significant attack vector for that data, making confidentiality a near-impossibility.

To solve this, we need a brand new approach to Vector Search - Confidential Vector Search. Cyborg Vector Search is the first solution to implement this. By leveraging cryptographic hashing and symmetric encryption, Cyborg Vector Search enables ANN search over encrypted space. This means that vector embeddings are **not** decrypted during search, and remain encrypted throughout their lifecycle. This greatly reduces the attack surface while guaranteeing the confidentiality of inference data.

**Why is it _important_?**

_Coming soon_

## How It Works

_Coming soon_

## Further Reading

To learn more about Cyborg Vector Search:

- [Deployment Models >](deployment-models.md)
- [Supported Backing Stores >](backing-stores.md)
- [Versions & Release Roadmap >](version-roadmap.md)
- [Security Breakdown (coming soon) >](security.md)

To get started now:

- [C++ API >](../cpp/)
- [Python API >](../python/)
- [Migration Guides (coming soon) >](../migration-guides/)
