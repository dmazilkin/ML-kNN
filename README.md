# k-Nearest Neighbors algorithm from scratch
This repository contains k-Nearest Neighbors (kNN) implementation from scratch.

## About
- Supported distance metrics:
  - **Euclidean** (`euclidean`),
  - **Manhattan** (`manhattan`),
  - **Cosine** (`cosine`),
  - **Chebyshev** (`chebyshev`).
- Supported weighting options:
  - `uniform`: equal weight for all neighbors,
  - `distance`: weight inversely proportional to distance (1/d),
  - `rank`: weight based on neighbor's sorted position (1/rank).

## Dependencies
To install all required dependencies, execute the following command:
```console
pip install requirements.txt
```

## Usage
To start main script, execute the following command:
```console
python main.py
```