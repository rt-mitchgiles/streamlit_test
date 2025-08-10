import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Cycling Performance Analyzer", page_icon="🚴‍♂️")

# Load Strava API credentials
STRAVA_CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]

# --- SECTION 0: Username Input & Token Mapping ---
@st.cache_data
def load_user_mapping(path="user_mapping.json"):
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_data
def save_user_mapping(mapping, path="user_mapping.json"):
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)

mapping = load_user_mapping()
username = st.text_input("Enter your username to continue:")
if not username:
    st.info("Please enter your username to access your data.")
    st.stop()

if username not in mapping:
    st.error(f"Username '{username}' not found in mapping.")
    st.stop()

user_info = mapping[username]
athlete_id = user_info.get("athlete_id")
access_token = user_info.get("access_token")
refresh_token = user_info.get("refresh_token")
token_expires_at = user_info.get("token_expires_at", 0)

# --- SECTION 1: Authentication Helpers ---
def refresh_strava_token(refresh_token):
    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        },
        timeout=10
    )
    response.raise_for_status()
    return response.json()

@st.cache_data(show_spinner=False)
def get_valid_access_token(user_info):
    now_ts = int(datetime.utcnow().timestamp())
    if user_info.get("token_expires_at", 0) <= now_ts:
        creds = refresh_strava_token(user_info["refresh_token"])
        # update local mapping
        user_info.update({
            "access_token": creds["access_token"],
            "refresh_token": creds["refresh_token"],
            "token_expires_at": creds["expires_at"]
        })
        mapping[username] = user_info
        save_user_mapping(mapping)
    return user_info["access_token"]

# --- SECTION 2: Data Loading & Preprocessing ---
DATA_PATH = "data/all_intervals_data.csv"
@st.cache_data
def load_intervals_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.rename(columns={
        "Date": "date", "Avg Power": "avg_power", "Avg HR": "avg_hr",
        "Load": "tss", "Fitness": "ctl", "Fatigue": "atl", "Form": "tsb"
    }, inplace=True)
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df.groupby("week")[['avg_power','avg_hr','tss','ctl','atl','tsb']].mean().reset_index()
    return df, weekly

df, weekly_summary = load_intervals_data(DATA_PATH)

# --- SECTION 3: Fetch Strava Activities ---
def fetch_strava_activities(token, after=None, before=None):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"per_page": 200, "page": 1}
    if after:
        params["after"] = int(after.timestamp())
    if before:
        params["before"] = int(before.timestamp())
    all_activities = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_activities.extend(data)
        params["page"] += 1
    return pd.DataFrame(all_activities)

# --- SECTION 4: UI: Load & Display Activities ---
if st.button("Load Recent Activities"):
    with st.spinner("Fetching activities from Strava..."):
        token = get_valid_access_token(user_info)
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        activities_df = fetch_strava_activities(token, after=week_ago)

    if activities_df.empty:
        st.info("No recent activities found.")
    else:
        activities_df['act_date'] = pd.to_datetime(activities_df['start_date_local']).dt.date
        st.subheader("🚴 Recent Activities (Last 7 Days)")
        st.dataframe(activities_df)

        # New activities
        csv_dates = set(df['date'].dt.date)
        new_acts = activities_df[~activities_df['act_date'].isin(csv_dates)]
        if not new_acts.empty:
            st.subheader("🆕 New Activities (not in CSV)")
            st.dataframe(new_acts)
        else:
            st.info("No new activities since last data load.")

# --- SECTION 5: Weekly Summary & Trends ---
st.header(f"🚴‍♂️ Weekly Summary for {username}")
st.dataframe(weekly_summary)

st.subheader("📈 Weekly Training Trends")
fig, ax = plt.subplots()
for metric in ["tss", "ctl", "atl", "tsb"]:
    ax.plot(weekly_summary['week'], weekly_summary[metric], label=metric.upper(), marker='o')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylabel("Score")
ax.set_title("Training Load Metrics")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# --- SECTION 6: GPT-4 Plan Generation ---
st.header("🧠 GPT-4 Performance Analysis and Training Plan")
if st.button("Generate 4-Week Plan"):
    summary_csv = weekly_summary.to_csv(index=False)
    prompt = (
        f"You are a professional cycling coach analyzing recent training performance for user: {username}.\n"
        "Weekly CSV data:\n"
        f"{summary_csv}\n"
        "1. Assess performance, fatigue, fitness balance.\n"
        "2. Create a 4-week cycling plan (3–5 workouts/week). Include Date,Name,Duration,Intensity,Notes in CSV format only."
    )
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional cycling coach."},
            {"role": "user", "content": prompt}
        ]
    )
    plan_csv = response.choices[0].message.content
    st.text_area("Your 4-Week Training Plan (CSV)", plan_csv, height=300)
    st.session_state['plan_csv'] = plan_csv

# --- SECTION 7: Upload Plan to Intervals.icu ---
st.header("🔗 Upload Plan to Intervals.icu")
plan_input = st.text_area("Paste the CSV plan here to upload")
if st.button("Upload Plan"):
    if not plan_input.strip():
        st.warning("Please paste your plan CSV above.")
    else:
        try:
            plan_df = pd.read_csv(StringIO(plan_input))
            plan_df["Date"] = pd.to_datetime(plan_df["Date"]).dt.strftime("%Y-%m-%d")
            payload = [
                {"date": r.Date, "name": r.Name, "duration": int(r.Duration), "intensity": float(r.Intensity), "description": r.Notes}
                for r in plan_df.itertuples()
            ]
            resp = requests.post(
                f"https://intervals.icu/api/v1/athlete/{athlete_id}/calendar",
                headers={"Authorization": f"Bearer {st.secrets['intervals_api_key']}", "Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            st.success("Plan uploaded successfully!")
        except Exception as e:
            st.error(f"Upload error: {e}")
