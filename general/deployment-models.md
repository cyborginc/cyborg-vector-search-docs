# Cyborg Vector Search - Deployment Models

Cyborg Vector Search can be deployed in four different ways:

## 1. Embedded Library

In this form, Cyborg Vector Search is packaged as a standalone module (Python or C++) which contains all of the encrypted indexing and querying logic. The module is run locally and can connect to any of the supported [backing stores](backing-stores.md).

**Recommended for evaluation, small-scale deployments, and custom integrations.**

## 2. Microservice (Docker Image)

_Coming Q1 2025_

The Cyborg Vector Search microservice is a self-contained Docker image that can be deployed locally or in a Kubernetes cluster. It enables scalable and repeatable deployments, and also connects to any of the supported [backing stores](backing-stores.md). REST API as well as client SDKs (Python, JS/TS, C++ and Go) will be available.

**Recommended for development and production deployments.**

## 3. Microservice w/ DB (Docker Image)

_Coming Mid-2025_

This evolved microservice integrates a backing store within the Docker image, enabling a self-sufficient Confidential Vector DB deployment without external backing stores. REST API as well as client SDKs (Python, JS/TS, C++ and Go) will be available.

**Recommended for development and production deployments.**

## 4. Serverless

_Release Date TBD_

The fully-managed Cyborg Vector Search offering will provide the lowest barrier to adoption - simply generate an API key and install a client SDK.

**Recommended for everything from evaluation to large-scale deployments.**