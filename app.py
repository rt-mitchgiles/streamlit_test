import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Cycling Performance Analyzer", layout="wide")

st.title("🚴 Cycling Performance Analyzer")
st.markdown("Analyzing your Intervals.icu export CSV using GPT-4o.")

# --- API KEY SETUP ---
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("API key required to proceed.")
    st.stop()

openai.api_key = openai_api_key

# --- LOAD DATA FROM FILE ---
DATA_PATH = "data/all_intervals_data.csv"

try:
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- RENAME AND CLEAN ---
df = df.rename(columns={
    'Date': 'date',
    'Avg HR': 'avg_hr',
    'Avg Power': 'avg_power',
    'Load': 'tss',
    'Fitness': 'ctl',
    'Fatigue': 'atl',
    'Form': 'tsb'
})

# Filter and drop missing essential values
df = df[['date', 'avg_power', 'avg_hr', 'tss', 'ctl', 'atl', 'tsb']].dropna()
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

# --- WEEKLY SUMMARY ---
weekly_summary = df.groupby('week').agg({
    'avg_power': 'mean',
    'avg_hr': 'mean',
    'tss': 'sum',
    'ctl': 'mean',
    'atl': 'mean',
    'tsb': 'mean'
}).reset_index()

# --- PLOTTING ---
st.subheader("📈 Weekly Training Summary")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weekly_summary['week'], weekly_summary['avg_power'], label='Avg Power (W)')
ax.plot(weekly_summary['week'], weekly_summary['avg_hr'], label='Avg HR (bpm)')
ax.plot(weekly_summary['week'], weekly_summary['tss'], label='TSS')
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
Based on this weekly training summary, provide a detailed analysis of the athlete's training progress, performance trends, fatigue and form, and recovery status. Include any signs of overtraining or undertraining and suggest recommendations for improvement or adjustment.

Each row contains:
- Week start date
- avg_power: Average power output (W)
- avg_hr: Average heart rate (bpm)
- tss: Training Stress Score
- ctl: Chronic Training Load (fitness)
- atl: Acute Training Load (fatigue)
- tsb: Training Stress Balance (form)

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