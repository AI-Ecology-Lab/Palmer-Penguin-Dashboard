import streamlit as st
from utils import load_data, set_page_config

def main():
    set_page_config()
    
    st.title("Palmer Penguins Analysis Dashboard")
    
    # Load and cache data
    df = st.session_state.get('data')
    if df is None:
        df = load_data()
        st.session_state['data'] = df
    
    # Display main page content
    st.write("Welcome to the Palmer Penguins Analysis Dashboard!")
    st.write("Select a page from the sidebar to explore different analyses.")

    # Display sample data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

if __name__ == "__main__":
    main()
