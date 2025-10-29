# 🛒 Recommendation System for Retail

This project explores and compares several recommendation models to suggest the **top 3 products** a customer is likely to purchase on their next visit to a retail store, such as **Leroy Merlin**. The models range from simple baselines to advanced graph-based approaches, and are evaluated using both **ranking metrics** and **basket-based evaluation**.

---
## Data Description

This project uses the **Online Retail dataset**, adapted to represent a retail store similar to Leroy Merlin. The dataset contains transactional records from an online retail business based in the United Kingdom, with no physical store.

- The dataset includes purchases from **2,725 customers** and **789 unique items**.
- Transactions cover the period from **December 1, 2010** to **December 9, 2011**.
- The evaluation focuses on the last three months of data, from **September 1, 2011** to **December 9, 2011**.

These customer-item transactions form the basis for building and evaluating the recommendation models in this project.

# Project File Structure and Description

## baseline/
- `cluster_based_recommendation.py`  
  Contains modular classes and functions implementing a cluster-based recommendation system. It handles customer clustering after feature engineering and generates top product recommendations per cluster.
- `matrix_factorization.py`  
  Provides a modular matrix factorization implementation used as the baseline model. Includes classes for training, prediction, and evaluation.

## graph_models/
- `bipart_graph_construction.py`  
  Defines reusable functions and classes for constructing the bipartite graph representing customer-product interactions, used as input for graph models.
- `GAT_model.py`  
  Implements the Graph Attention Network model in a modular way, including classes for model architecture, training loops, and inference.
- `GCN_model.py`  
  Contains modular implementations of Graph Convolutional Network components for recommendation, encapsulated as classes and helper functions.
- `weightedGCN_model.py`  
  Extends GCN with weighted edges; includes classes and methods to build and train this enhanced graph model.

## evaluation/
- `evaluation_metrics.py`  
  Provides a collection of functions and classes to compute ranking metrics (Recall@3, NDCG@3, MAP@3) and basket-based metrics (Hit Rate, Lift, statistical tests). Designed for flexible evaluation workflows.

## data/
- `feature_engineering.py`  
  Includes modular functions and classes for feature extraction, transformation, and scaling tailored to the retail dataset.
- `retail_preprocessing.py`  
  Contains preprocessing pipelines implemented as classes and functions to clean and prepare raw retail data for modeling.

## experiments/
- `LeroyMerline_recommendation_model.ipynb`  
  A comprehensive Jupyter notebook orchestrating data processing, model training, evaluation, and visualization using the modular codebase.
- `model_comparaison.docx`  
  A written report summarizing model performances, comparisons, and insights from the experiments.




## 🧠 Models Compared

### 🔹 Baseline Models
- **Cluster-Based Recommender**: Clusters customers using engineered features and recommends the most popular items within each cluster.
- **Matrix Factorization (MF)**: Standard collaborative filtering model used as a baseline.

### 🔹 Graph-Based Models
- **GCN (Graph Convolutional Network)**: Leverages the structure of the customer-product interaction graph.
- **GAT (Graph Attention Network)**: Uses attention mechanisms to weigh neighbor importance.
- **Weighted GCN**: Enhances GCN with edge weights to better model purchase frequency or recency.

---

## 📊 Evaluation

We use two main categories of metrics:

### 1. **Ranking-Based Metrics**
Evaluates the quality of the top-3 recommendations:
- **Recall@3**
- **NDCG@3**
- **MAP@3**
- **Lift@3**

### 2. **Basket-Based Metrics**
Evaluates if at least one of the recommended items is purchased:
- **Hit Rate**
- **Baseline Rate**
- **Lift**
- **p-value (Statistical Significance)**

> 📈 Graph-based models, especially **Weighted GCN**, show significant improvement over the baseline MF in all ranking metrics (e.g., +121% Recall@3 gain), proving the benefit of leveraging graph structure.

---

## 📌 Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- PyTorch, DGL or PyTorch Geometric (for graph models)
- matplotlib, seaborn (for visualization)

> You can install the required packages via:
```bash
pip install -r requirements.txt
