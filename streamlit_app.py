import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Function to load and preprocess data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to preprocess data
def preprocess_data(data):
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data, data_scaled

# Function to apply DBSCAN
def apply_dbscan(data, eps, min_samples):
    data_preprocessed, data_scaled = preprocess_data(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data_scaled)
    data_preprocessed['DBSCAN_Cluster'] = dbscan_labels
    return data_preprocessed, dbscan_labels, data_scaled

# Function to apply Hierarchical Clustering
def apply_hierarchical(data, n_clusters):
    data_preprocessed, data_scaled = preprocess_data(data)
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    hc_labels = hc.fit_predict(data_scaled)
    data_preprocessed['HC_Cluster'] = hc_labels
    return data_preprocessed, hc_labels, data_scaled

# Streamlit app layout
st.title('Clustering with DBSCAN and Hierarchical Clustering')

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # DBSCAN parameters
    st.sidebar.header('DBSCAN Parameters')
    eps = st.sidebar.slider('Epsilon (eps)', 0.1, 10.0, 0.5)
    min_samples = st.sidebar.slider('Minimum Samples', 1, 10, 5)
    
    # Hierarchical Clustering parameters
    st.sidebar.header('Hierarchical Clustering Parameters')
    n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 3)

    if st.sidebar.button('Apply DBSCAN'):
        data_dbscan, dbscan_labels, data_scaled = apply_dbscan(data, eps, min_samples)
        st.write("DBSCAN Clustering Results:")
        st.write(data_dbscan[['DBSCAN_Cluster']].value_counts())

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data_dbscan['DBSCAN_Cluster'], palette='viridis')
        plt.title('DBSCAN Clusters')
        st.pyplot(plt)

    if st.sidebar.button('Apply Hierarchical Clustering'):
        data_hc, hc_labels, data_scaled = apply_hierarchical(data, n_clusters)
        st.write("Hierarchical Clustering Results:")
        st.write(data_hc[['HC_Cluster']].value_counts())

        # Plot dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram = sch.dendrogram(sch.linkage(data_scaled, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Euclidean distances')
        st.pyplot(plt)

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data_hc['HC_Cluster'], palette='viridis')
        plt.title('Hierarchical Clustering Clusters')
        st.pyplot(plt)
