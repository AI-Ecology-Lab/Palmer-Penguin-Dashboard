import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Get data from session state
df = st.session_state['data']

st.title("Penguin Clustering Analysis")

# Select features for clustering
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
n_clusters = st.slider("Number of Clusters", 2, 6, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df_cluster = df.copy()
df_cluster['Cluster'] = clusters

# Create 3D scatter plot
st.subheader("3D Cluster Visualization")
fig = px.scatter_3d(
    df_cluster,
    x='bill_length_mm',
    y='bill_depth_mm',
    z='flipper_length_mm',
    color='Cluster',
    hover_data=['species'],
    title='Penguin Clusters'
)
st.plotly_chart(fig)

# Show cluster statistics
st.subheader("Cluster Statistics")
for i in range(n_clusters):
    st.write(f"Cluster {i} Statistics:")
    cluster_data = df_cluster[df_cluster['Cluster'] == i]
    st.write(cluster_data[features].describe())
