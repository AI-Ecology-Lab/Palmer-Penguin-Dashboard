import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Get data from session state
df = st.session_state['data']

st.title("Palmer Penguins Dataset Overview")

# Display basic dataset information
st.subheader("Dataset Information")
st.write(f"Number of records: {len(df)}")
st.write(f"Number of features: {len(df.columns)}")

# Show sample distribution
st.subheader("Sample Distribution")
col1, col2 = st.columns(2)

with col1:
    fig = px.pie(df, names='species', title='Species Distribution')
    st.plotly_chart(fig)

with col2:
    fig = px.pie(df, names='island', title='Island Distribution')
    st.plotly_chart(fig)

# Correlation heatmap
st.subheader("Feature Correlations")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
