import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

# --- SECTION 0: Username Input Template ---
# Simple page to enter a username and tailor content accordingly
st.set_page_config(page_title="Cycling Performance Analyzer", page_icon="🚴‍♂️")
st.title("🔑 Welcome to the Cycling Performance Analyzer")

@st.cache_data
def load_user_mapping(path="user_mapping.json"):
    with open(path, 'r') as f:
        return json.load(f)

mapping = load_user_mapping()

username = st.text_input("Enter your username to continue")
if not username:
    st.info("Please enter your username above to access your data.")
    st.stop()

athlete_id = mapping.get(username)
if athlete_id is None:
    st.error(f"Username '{username}' not found.")
    st.stop()

# --- SECTION 1: Load & Preprocess Data ---
DATA_PATH = "data/all_intervals_data.csv"
DATE_FORMAT = "%Y-%m-%d"
openai_api_key = st.secrets["OPENAI_API_KEY"]
icu_api_key = st.secrets["intervals_api_key"]
# athlete_id is now from mapping

@st.cache_data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.rename(columns={
        "Date": "date",
        "Avg Power": "avg_power",
        "Avg HR": "avg_hr",
        "Load": "tss",
        "Fitness": "ctl",
        "Fatigue": "atl",
        "Form": "tsb"
    })
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df.groupby("week")[['avg_power','avg_hr','tss','ctl','atl','tsb']].mean().reset_index()
    return df, weekly

# Load data for this user
# (In real scenario, filter df by athlete_id if multi-user file)
df, weekly_summary = load_and_preprocess(DATA_PATH)

# --- SECTION 2: Display Data & Trends ---
st.header(f"🚴‍♂️ Weekly Summary for {username}")
st.dataframe(weekly_summary)

st.subheader("📈 Weekly Training Trends")
fig, ax = plt.subplots()
for metric in ["tss", "ctl", "atl", "tsb"]:
    ax.plot(weekly_summary["week"], weekly_summary[metric], label=metric.upper(), marker='o')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylabel("Score")
ax.set_title("Training Load Metrics")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# --- SECTION 3: GPT-4o Analysis & Plan ---
st.header("🧠 GPT-4o Performance Analysis and Training Plan")
if st.button("Generate 4-Week Plan"):
    summary_csv = weekly_summary.to_csv(index=False)
    prompt = f"""
You are a professional cycling coach analyzing recent training performance for user: {username}.

Weekly CSV data:
{summary_csv}

1. Assess performance, fatigue, fitness balance.
2. Create a 4-week cycling plan (3–5 workouts/week).
Include Date,Name,Duration,Intensity,Notes in CSV format only.
"""
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a professional cycling coach."},
            {"role":"user","content":prompt}
        ]
    )
    plan_csv = response.choices[0].message.content
    st.text_area("Your 4-Week Training Plan (CSV)", plan_csv, height=200)
    st.session_state['plan_csv'] = plan_csv

# --- SECTION 4: Upload to Intervals.icu ---
st.header("🔗 Upload Plan to Intervals.icu")
plan_input = st.text_area("Paste the CSV plan here to upload")
if st.button("Upload Plan"):
    if not plan_input.strip():
        st.warning("Please paste your plan CSV above.")
    else:
        try:
            plan_df = pd.read_csv(StringIO(plan_input))
            plan_df["Date"] = pd.to_datetime(plan_df["Date"]).dt.strftime(DATE_FORMAT)
            payload = [{
                "date": r.Date,
                "name": r.Name,
                "duration": int(r.Duration),
                "intensity": float(r.Intensity),
                "description": r.Notes
            } for r in plan_df.itertuples()]
            resp = requests.post(
                f"https://intervals.icu/api/v1/athlete/{athlete_id}/calendar",
                headers={"Authorization": f"Bearer {icu_api_key}", "Content-Type": "application/json"},
                json=payload
            )
            if resp.ok:
                st.success("Plan uploaded successfully!")
            else:
                st.error(f"Upload failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
