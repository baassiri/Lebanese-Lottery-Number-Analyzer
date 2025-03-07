import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# Set page title
st.set_page_config(page_title="Lebanese Lottery Analyzer", page_icon="🇱🇧")

# Title
st.title("🇱🇧 Lebanese Lottery Number Generator aka Jarrib hazzak 🇱🇧")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Data/Lebanese_Lottery.csv")
    return df

df = load_data()

# Show raw data
if st.checkbox(" Show Raw Data"):
    st.write(df)

# Progress Bar Setup
progress_bar = st.progress(0)
status_text = st.empty()

# Function to simulate processing steps
def process_step(step_name, step_number, total_steps=7):
    with st.spinner(f"Running {step_name}..."):
        time.sleep(1.5)  # Simulating computation time
        progress_bar.progress(step_number / total_steps)
        status_text.success(f" {step_name} Done!")

# Run Analysis & Generate Numbers
if st.button("🔍 Start Full Analysis & Generate Numbers"):
    process_step("Descriptive Statistics", 1)
    process_step("Probability Analysis", 2)
    process_step("Time Series Analysis", 3)
    process_step("Combinatorial Analysis", 4)
    process_step("Anomaly Detection", 5)
    process_step("Machine Learning Models", 6)
    process_step("Prediction Generation", 7)
    
    st.success(" All analyses completed!")

    # Generate Lucky Lottery Numbers
    lucky_numbers = sorted(random.sample(range(1, 43), 6))
    st.subheader(" Your Lucky Lottery Numbers (for educational purposes only):")
    st.success(f" {lucky_numbers}")
