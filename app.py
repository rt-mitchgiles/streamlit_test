import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Cycling Performance Analyzer", layout="wide")

st.title("🚴 Cycling Performance Analyzer")
st.markdown("Upload your training data to get performance insights using GPT-4o.")

# --- API KEY SETUP ---
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("API key required to proceed.")
    st.stop()

openai.api_key = openai_api_key

# --- DATA UPLOAD ---
uploaded_file = st.file_uploader("Upload your training CSV (e.g., Intervals.icu export)", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file, parse_dates=['date'])

# --- BASIC CLEANING ---
if 'date' not in df.columns:
    st.error("CSV must contain a 'date' column.")
    st.stop()

# Select only relevant columns
df = df[['date', 'avg_power', 'avg_hr', 'tss', 'sleep_hours']].dropna()
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

# --- WEEKLY SUMMARY ---
weekly_summary = df.groupby('week').agg({
    'avg_power': 'mean',
    'avg_hr': 'mean',
    'tss': 'sum',
    'sleep_hours': 'mean'
}).reset_index()

# --- PLOTTING ---
st.subheader("📈 Weekly Training Summary")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weekly_summary['week'], weekly_summary['avg_power'], label='Avg Power (W)')
ax.plot(weekly_summary['week'], weekly_summary['avg_hr'], label='Avg HR (bpm)')
ax.set_ylabel("Value")
ax.set_xlabel("Week")
ax.legend()
st.pyplot(fig)

# --- GPT-4 ANALYSIS ---
st.subheader("🧠 AI Performance Analysis")

# Format summary as text
summary_text = weekly_summary.to_string(index=False)

prompt = f"""
You are a cycling coach and performance physiologist.
Based on this weekly training summary including average power, heart rate, TSS (Training Stress Score), and sleep hours, give a detailed analysis of the athlete's training progress, fatigue trends, and recovery status. Suggest what the athlete should focus on next.

Weekly data:
{summary_text}
"""

if st.button("Analyze with GPT-4o"):
    try:
        with st.spinner("Analyzing performance with GPT-4o..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a cycling performance expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.success("Analysis complete")
            st.markdown(response["choices"][0]["message"]["content"])
    except Exception as e:
        st.error(f"API error: {e}")
