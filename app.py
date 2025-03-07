import streamlit as st
import pandas as pd
import random

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("Data/Lebanese_Lottery.csv")

df = load_data()

# App Title
st.title("🎰 Lebanese Lottery Number Generator 🎰")

# Show dataset
if st.checkbox("Show Raw Data"):
    st.write(df)

# Generate Lucky Numbers
st.header("🔮 Generate Lucky Lottery Numbers")
if st.button("Click to Generate"):
    lucky_numbers = sorted(random.sample(range(1, 100), 6))
    st.success(f"Your lucky numbers: {lucky_numbers}")
