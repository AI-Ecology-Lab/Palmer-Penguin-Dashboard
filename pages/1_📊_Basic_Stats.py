import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get data from session state
df = st.session_state['data']

st.title("Basic Statistics")

# Display summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Create visualizations
st.subheader("Distribution of Numeric Features")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

col1, col2 = st.columns(2)
with col1:
    feature = st.selectbox("Select Feature", numeric_cols)
    
with col2:
    group_by = st.selectbox("Group by", ['None', 'species', 'sex', 'island'])

if group_by != 'None':
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=group_by, y=feature)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=feature)
    st.pyplot(fig)
