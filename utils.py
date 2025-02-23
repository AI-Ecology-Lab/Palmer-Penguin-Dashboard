import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv('data/palmerpenguins_extended.csv')
    return df

def set_page_config():
    st.set_page_config(
        page_title="Palmer Penguins Analysis",
        page_icon="🐧",
        layout="wide"
    )

def create_scatter_plot(df, x_col, y_col, color_col='species', title=None):
    fig = px.scatter(df, x=x_col, y=y_col, 
                    color=color_col,
                    title=title,
                    template='plotly_white')
    return fig

def create_box_plot(df, x_col, y_col, title=None):
    fig = px.box(df, x=x_col, y=y_col,
                 title=title,
                 template='plotly_white')
    return fig

def preprocess_features(df, features):
    scaler = StandardScaler()
    return scaler.fit_transform(df[features])

def create_correlation_heatmap(df, features):
    corr = df[features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=features,
        y=features,
        colorscale='RdBu'
    ))
    fig.update_layout(title='Feature Correlation Heatmap')
    return fig

def plot_feature_importance(importance, features, title):
    fig = px.bar(x=features, y=importance,
                 title=title,
                 labels={'x': 'Features', 'y': 'Importance'})
    return fig
