# k-Nearest Neighbors algorithm from scratch
This repository contains k-Nearest Neighbors (kNN) implementation from scratch for classification and regression problems.

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
python main.py [OPTIONS]
```

### Available options
- **-e, --example** (required) - type of example to run. Available examples: regression, classification.
- **-t, --train** (required) - size of training dataset.
- **-p, --predict** (required) - size of dataset to predict.
- **-k' (required) - count of nearest neighbors that are using for predicting.
- **-m, --metric** (optional) - distance metrics are used to determine the similarity between objects. Available metrics: euclidean, manhattain, chebyshev, cosine. Default metric is euclidean.
- **-w, --weight** (optional) - set weight for weighted kNN. Defalut equal weight for all neighbors is set.