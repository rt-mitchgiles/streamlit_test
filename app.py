import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- SECTION 0: Built-in Authentication (no extra packages) ---
st.set_page_config(page_title="Cycling Performance Analyzer", page_icon="🚴‍♂️")

# Load credentials from Streamlit secrets (define in .streamlit/secrets.toml)
# Example secrets.toml:
# username = "mitchell"
# password = "securepass"
USERNAME = st.secrets.get("username")
PASSWORD = st.secrets.get("password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# --- SECTION 1: Load and preprocess the data ---
DATA_PATH = "data/all_intervals_data.csv"
DATE_FORMAT = "%Y-%m-%d"

openai_api_key = st.secrets["OPENAI_API_KEY"]
icu_api_key = st.secrets["intervals_api_key"]
athlete_id = st.secrets["athlete_id"]

@st.cache_data
def load_and_preprocess_data(file_path):
    # Load raw data
    df = pd.read_csv(file_path, parse_dates=["Date"])
    # Rename for convenience
    df = df.rename(columns={
        "Date": "date",
        "Avg Power": "avg_power",
        "Avg HR": "avg_hr",
        "Load": "tss",
        "Fitness": "ctl",
        "Fatigue": "atl",
        "Form": "tsb"
    })
    # Compute week start
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    # Aggregate weekly means
    weekly = df.groupby("week")[['avg_power','avg_hr','tss','ctl','atl','tsb']].mean().reset_index()
    return df, weekly

# Load data
df, weekly_summary = load_and_preprocess_data(DATA_PATH)

# --- SECTION 2: Display data and metrics ---
st.title("🚴‍♂️ Cycling Performance Analyzer with GPT-4o")

st.subheader("📊 Weekly Summary")
st.dataframe(weekly_summary)

st.subheader("📈 Weekly Training Trends")
fig, ax = plt.subplots()
ax.plot(weekly_summary["week"], weekly_summary["tss"], label="TSS", marker='o')
ax.plot(weekly_summary["week"], weekly_summary["ctl"], label="CTL", marker='o')
ax.plot(weekly_summary["week"], weekly_summary["atl"], label="ATL", marker='o')
ax.plot(weekly_summary["week"], weekly_summary["tsb"], label="TSB", marker='o')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend()
ax.set_ylabel("Score")
ax.set_title("Training Load Metrics")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- SECTION 3: GPT-4o Analysis ---
st.subheader("🧠 GPT-4o Performance Analysis and Training Plan")

run_analysis = st.button("Analyze and Generate Plan")
if run_analysis:
    client = OpenAI(api_key=openai_api_key)
    summary_csv = weekly_summary.to_csv(index=False)
    prompt = f"""
    You are a professional cycling coach analyzing recent training performance.

    You are given weekly cycling data:
    - avg_power: average power in watts
    - avg_hr: average heart rate in bpm
    - tss: Training Stress Score
    - ctl: Chronic Training Load (fitness)
    - atl: Acute Training Load (fatigue)
    - tsb: Training Stress Balance (form)

    Weekly training data (CSV format):
    {summary_csv}

    Based on this data:

    1. Briefly assess performance, fatigue, and fitness balance.
    2. Generate a structured 4-week cycling training plan with 3–5 workouts per week.
    3. For each workout, include Date (next Monday start), Name, Duration (min), Intensity (%FTP), Notes.

    Output as CSV with headers: Date,Name,Duration,Intensity,Notes. No markdown.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a professional cycling coach."},
            {"role":"user","content":prompt}
        ]
    )
    plan_csv = response.choices[0].message.content
    st.markdown("### 📋 GPT-4o Training Plan")
    st.code(plan_csv, language='csv')
    st.session_state["gpt_plan_raw"] = plan_csv

# --- SECTION 4: Upload to Intervals.icu ---
st.subheader("🔗 Upload Plan to Intervals.icu")
with st.expander("Paste the GPT-generated plan CSV here to upload"):
    user_pasted_table = st.text_area("CSV table")
if st.button("Upload to Intervals.icu"):
    if not user_pasted_table:
        st.warning("Paste the plan first!")
    else:
        try:
            plan_df = pd.read_csv(StringIO(user_pasted_table))
            plan_df["Date"] = pd.to_datetime(plan_df["Date"]).dt.strftime(DATE_FORMAT)
            payload = [
                {"date":r.Date, "name":r.Name, "duration":int(r.Duration),
                 "intensity":float(r.Intensity), "description":r.Notes}
                for r in plan_df.itertuples()
            ]
            resp = requests.post(
                f"https://intervals.icu/api/v1/athlete/{athlete_id}/calendar",
                headers={"Authorization":f"Bearer {icu_api_key}","Content-Type":"application/json"},
                json=payload
            )
            if resp.ok:
                st.success("Plan uploaded successfully!")
            else:
                st.error(f"Upload failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Error parsing/uploading plan: {e}")
