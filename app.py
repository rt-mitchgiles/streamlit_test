import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from streamlit_authenticator import Authenticate
import yaml

# --- SECTION 0: Easy Authentication ---
st.set_page_config(page_title="Cycling Performance Analyzer", page_icon="🚴‍♂️")

# Define users (for demo; in production use env vars or db)
USERS = [
    {'name': 'Mitchell Giles', 'username': 'mitchell', 'password': 'pass123'},
    {'name': 'Demo User', 'username': 'user', 'password': 'pass'}
]

# Create YAML config (in-memory for demo)
config = {
    'credentials': {
        'usernames': {
            u['username']: {'name': u['name'], 'password': u['password']} for u in USERS
        }
    },
    'cookie': {
        'expiry_days': 1,
        'key': 'cycling-auth',
        'name': 'cycling-auth-cookie'
    },
    'preauthorized': {
        'emails': []
    }
}

# Place authenticator.login OUTSIDE of 'with st.sidebar:'
st.sidebar.title('User Login')
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
# This must be outside any sidebar block! Location is 'sidebar' so authenticator manages sidebar itself
name, authentication_status, username = authenticator.login('Login', 'sidebar')

if not authentication_status:
    if authentication_status is False:
        st.error('Invalid username or password')
    st.stop()

# --- SECTION 1: Load and preprocess the data ---
DATA_PATH = "data/all_intervals_data.csv"
DATE_FORMAT = "%Y-%m-%d"

openai_api_key = st.secrets["OPENAI_API_KEY"]
icu_api_key = st.secrets["intervals_api_key"]
athlete_id = st.secrets["athlete_id"]

@st.cache_data
def load_and_preprocess_data(file_path):
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
    weekly = df.groupby("week")[["avg_power", "avg_hr", "tss", "ctl", "atl", "tsb"]].mean().reset_index()
    return df, weekly

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

if run_analysis and openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    # Convert summary to string table
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

    1. Briefly assess the user's performance, fatigue and fitness balance.
    2. Generate a structured **4-week cycling training plan** with 3–5 workouts per week.
    3. For each workout, include the following fields:
    - `Date` (starting next Monday)
    - `Name` (e.g. "Endurance Ride", "Sweet Spot Intervals")
    - `Duration` (in minutes)
    - `Intensity` (as % of FTP)
    - `Notes` (short workout description)

    **Important**: Output the training plan as a valid comma-separated table (CSV format), with no Markdown formatting or bullet points. Use these headers exactly:

    `Date,Name,Duration,Intensity,Notes`

    Start with the training plan immediately. Do not explain or describe it — only output the table.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional cycling coach."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_reply = response.choices[0].message.content
    st.markdown("### 📋 GPT-4o Analysis and Plan")
    st.markdown(gpt_reply)

    st.session_state["gpt_plan_raw"] = gpt_reply

# --- SECTION 4: Upload to Intervals.icu ---
st.subheader("🔗 Upload Plan to Intervals.icu")

with st.expander("Paste in the GPT-generated training plan table if it appeared above"):
    user_pasted_table = st.text_area("Paste the full table here (Markdown or CSV-style)")

def parse_training_plan(text):
    try:
        table_io = StringIO(text)
        plan_df = pd.read_csv(table_io)
        plan_df.columns = [col.strip() for col in plan_df.columns]
        plan_df["Date"] = pd.to_datetime(plan_df["Date"]).dt.strftime(DATE_FORMAT)
        return plan_df
    except Exception as e:
        st.error(f"Failed to parse training plan: {e}")
        return None

def upload_to_intervals_icu(plan_df, api_key, athlete_id):
    url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/calendar"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = []
    for _, row in plan_df.iterrows():
        payload.append({
            "date": row["Date"],
            "name": row["Name"],
            "duration": int(row["Duration"]),
            "intensity": float(row["Intensity"]),
            "description": row["Notes"]
        })
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return "✅ Successfully uploaded the training plan!"
    else:
        return f"❌ Upload failed: {response.status_code} — {response.text}"

if st.button("📤 Upload to Intervals.icu"):
    if not user_pasted_table:
        st.warning("You need to paste the GPT training table above first.")
    elif not icu_api_key or not athlete_id:
        st.warning("You must enter your Intervals.icu API key and Athlete ID.")
    else:
        plan_df = parse_training_plan(user_pasted_table)
        if plan_df is not None:
            result = upload_to_intervals_icu(plan_df, icu_api_key, athlete_id)
            st.success(result)
