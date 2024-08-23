# Clustering and Vaccination Optimization Project

### Project Overview

This repository includes various clustering algorithms and optimization methods to address clustering challenges and support the initialization phase of the "Vaccine for All" project. The project is intended to develop equitable and efficient vaccination distribution models, particularly in large-scale vaccination campaigns like those necessitated by the COVID-19 pandemic.

This repository consists of several Python scripts implementing different clustering techniques, warm start strategies, and a distance matrix-based Gurobi optimization approach. The results of these methods are crucial for simulating and optimizing vaccination strategies, ensuring vaccines reach targeted groups efficiently.

### Repository Structure

- **20 Clusters.xlsx**: Contains the 2D coordinates used for visualization and clustering in various scripts.
- **20pts.txt**: Provides additional data points used in clustering tasks.
- **LICENSE**: The project's license file.
- **dbscan.py**: Implements the DBSCAN clustering algorithm.
- **djlorj.txt**: Contains the distance matrix used for optimization and clustering processes.
- **gmm.py**: Implements the Gaussian Mixture Model (GMM) clustering algorithm.
- **hierarchical_agglomerative.py**: Contains the hierarchical agglomerative clustering algorithm and its visualization.
- **kmeans.py**: Implements the K-means clustering algorithm.
- **main.py**: Runs the primary clustering and optimization processes with warm starts.
- **main_coord_fullwarm.py**: Implements Gurobi optimization using full warm start with 2D coordinates.
- **main_coord_partialwarm.py**: Implements Gurobi optimization using partial warm start with 2D coordinates.
- **main_dist_fullwarm.py**: Implements Gurobi optimization using full warm start with a distance matrix.
- **main_dist_partialwarm.py**: Implements Gurobi optimization using partial warm start with a distance matrix.
- **preprocessing.py**: Handles data preprocessing, including cleaning the distance matrix.

### Clustering Methods

1. **K-means Clustering (`kmeans.py`)**
   - Clusters data points based on their 2D coordinates.
   - Uses elbow method to determine the optimal number of clusters.

2. **Gaussian Mixture Model (`gmm.py`)**
   - Implements a probabilistic clustering approach using Gaussian distributions.
   - Supports soft clustering, allowing each point to belong to multiple clusters with different probabilities.

3. **DBSCAN (`dbscan.py`)**
   - A density-based clustering algorithm used to identify clusters of high density in data.
   - Particularly useful for data with noise and outliers.

4. **Hierarchical Agglomerative Clustering (`hierarchical_agglomerative.py`)**
   - Performs agglomerative clustering, where each data point starts as its cluster, and clusters are successively merged.
   - A dendrogram is used to visualize the hierarchical structure of clusters.

### Optimization with Gurobi

1. **Full Warm Start with 2D Coordinates (`main_coord_fullwarm.py`)**
   - Utilizes Gurobi for optimization, starting with K-means clusters.
   - Optimization aims to minimize the difference between maximum and average distances within clusters.

2. **Partial Warm Start with 2D Coordinates (`main_coord_partialwarm.py`)**
   - Similar to full warm start, but only a subset of data points are used for the warm start.

3. **Full Warm Start with Distance Matrix (`main_dist_fullwarm.py`)**
   - Performs optimization using a precomputed distance matrix, with a full warm start strategy.

4. **Partial Warm Start with Distance Matrix (`main_dist_partialwarm.py`)**
   - Optimizes clusters using a distance matrix with a partial warm start strategy.
   - Uses Gurobi's optimization capabilities to find the best cluster assignments under time constraints.

### Integration with the "Vaccine for All" Project

This clustering and optimization project serves as a crucial initialization step for the "Vaccine for All" project, which is focused on equitable vaccine distribution. The models and methods developed here are designed to support real-world vaccination logistics by optimizing vaccine distribution to targeted groups and underserved populations.

### How to Use

1. **Install Dependencies**
   - Ensure you have Python 3 installed.
   - Install required libraries using pip:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Clustering Algorithms**
   - Execute any of the clustering scripts (e.g., `kmeans.py`, `hierarchical_agglomerative.py`) to perform clustering on the data provided in `20 Clusters.xlsx`.

3. **Run Gurobi Optimization**
   - Execute `main_coord_fullwarm.py`, `main_coord_partialwarm.py`, `main_dist_fullwarm.py`, or `main_dist_partialwarm.py` depending on the desired clustering strategy and data input.

4. **Visualization**
   - The scripts include visualization of clusters using matplotlib, showing how points are grouped in 2D space.

### Contribution and License

This project is open to contributions. Please ensure that any new features or bug fixes are well-documented and tested. The project is licensed under the MIT License, as detailed in the LICENSE file.

### Acknowledgments

This project is developed in the context of the "Vaccine for All" project, led by Prof. Dr. Sibel Salman, with support from the Katip Çelebi-Newton Fund and TÜBİTAK. The methods developed here are part of the ongoing research to optimize vaccine distribution in response to pandemics like COVID-19.
