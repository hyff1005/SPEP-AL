import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def pca_embeddings(embeddings_df, n_components=10):
    if embeddings_df is None:
        return None

    indices = embeddings_df.index

    numeric_df = embeddings_df.select_dtypes(include=[np.number])

    if numeric_df.isnull().values.any():
        imputer = SimpleImputer(strategy='mean')
        numeric_data = imputer.fit_transform(numeric_df)
    else:
        numeric_data = numeric_df.values

    n_samples, n_features = numeric_data.shape
    adj_n_components = min(n_components, n_samples, n_features)
    
    if adj_n_components < n_components:
        print(f"Warning: n_components adjusted from {n_components} to {adj_n_components} due to data constraints.")

    pca = PCA(n_components=adj_n_components)
    embeddings_pca = pca.fit_transform(numeric_data)

    embeddings_pca_df = pd.DataFrame(
        embeddings_pca,
        index=indices,
        columns=[f"PCA {i}" for i in range(1, adj_n_components + 1)],
    )

    return embeddings_pca_df