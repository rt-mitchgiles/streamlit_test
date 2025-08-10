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

# --- SECTION 0: Username Input & First-Time OAuth Flow ---
st.set_page_config(page_title="Cycling Performance Analyzer", page_icon="🚴‍♂️")
st.title("🔑 Welcome to the Cycling Performance Analyzer")

def load_user_mapping(path="user_mapping.json"):
    with open(path, 'r') as f:
        return json.load(f)
mapping = load_user_mapping()

# 1) Input username
username = st.text_input("Enter your username to continue")
if not username:
    st.info("Please enter your username above to access your data.")
    st.stop()

# 2) Look up user_info
user_info = mapping.get(username)
if user_info is None:
    st.error(f"Username '{username}' not found in mapping.")
    st.stop()

# 3) If no refresh_token, start OAuth flow
gosed_token = user_info.get("refresh_token")
if not gosed_token:
    params = {
        "client_id":      st.secrets["STRAVA_CLIENT_ID"],
        "response_type":  "code",
        "redirect_uri":   st.secrets["STRAVA_REDIRECT_URI"],
        "approval_prompt":"auto",
        "scope":          "activity:read_all"
    }
    auth_url = "https://www.strava.com/oauth/authorize?" + "&".join(f"{k}={v}" for k,v in params.items())
    st.markdown(f"[👉 Authorize with Strava]({auth_url})")

    qs = st.experimental_get_query_params()
    if "code" in qs:
        code = qs["code"][0]
        try:
            resp = requests.post(
                "https://www.strava.com/oauth/token",
                data={
                    "client_id":     st.secrets["STRAVA_CLIENT_ID"],
                    "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
                    "code":          code,
                    "grant_type":    "authorization_code"
                },
                timeout=10
            )
            resp.raise_for_status()
            creds = resp.json()
            user_info.update({
                "access_token":     creds["access_token"],
                "refresh_token":    creds["refresh_token"],
                "token_expires_at": creds["expires_at"],
                "athlete_id":       creds["athlete"]["id"]
            })
            st.success("✅ Strava connected! Reload the app to fetch activities.")
        except requests.HTTPError as err:
            st.error(f"OAuth exchange failed: {err.response.text}")
    st.stop()

# 4) Now read tokens for normal path
athlete_id       = user_info["athlete_id"]
access_token     = user_info["access_token"]
refresh_token    = user_info["refresh_token"]
token_expires_at = user_info["token_expires_at"]

# --- SECTION 1: Load & Preprocess CSV Data ---
DATA_PATH = "data/all_intervals_data.csv"
DATE_FORMAT = "%Y-%m-%d"
openai_api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Avg Power": "avg_power", "Avg HR": "avg_hr",
                             "Load": "tss", "Fitness": "ctl", "Fatigue": "atl", "Form": "tsb"})
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df.groupby("week")[['avg_power','avg_hr','tss','ctl','atl','tsb']].mean().reset_index()
    return df, weekly

df, weekly_summary = load_and_preprocess(DATA_PATH)

# --- SECTION 2: OAuth Helpers & Token Refresh ---
def refresh_strava_token(client_id, client_secret, refresh_token):
    url = "https://www.strava.com/oauth/token"
    data = {
        "client_id":     client_id,
        "client_secret": client_secret,
        "grant_type":    "refresh_token",
        "refresh_token": refresh_token
    }
    resp = requests.post(url, data=data, timeout=10)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        st.error(f"Strava token refresh failed [{resp.status_code}]: {resp.text}")
        raise
    return resp.json()


def get_valid_access_token(user_info):
    now_ts = int(datetime.utcnow().timestamp())
    expires = user_info.get('token_expires_at')
    if expires is None or now_ts > expires:
        try:
            creds = refresh_strava_token(
                st.secrets['STRAVA_CLIENT_ID'],
                st.secrets['STRAVA_CLIENT_SECRET'],
                user_info['refresh_token']
            )
        except Exception:
            st.stop()
        user_info.update({
            'access_token':     creds['access_token'],
            'refresh_token':    creds['refresh_token'],
            'token_expires_at': creds['expires_at']
        })
    return user_info['access_token']

# --- SECTION 3: Fetch & Display Activities ---
def fetch_strava_activities(access_token, after=None, before=None):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {'per_page':200, 'page':1}
    if after:  params['after']  = int(after.timestamp())
    if before: params['before'] = int(before.timestamp())
    all_acts = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_acts.extend(data)
        params['page'] += 1
    return pd.DataFrame(all_acts)

if st.button("Load Recent Activities"):
    with st.spinner("Fetching activities from Strava..."):
        valid_token = get_valid_access_token(user_info)
        week_ago    = datetime.utcnow() - timedelta(days=7)
        activities_df = fetch_strava_activities(valid_token, after=week_ago)
    if activities_df.empty:
        st.info("No activities found or error occurred.")
    else:
        st.subheader("🚴 Recent Activities (Last 7 Days)")
        st.dataframe(activities_df)
        activities_df['act_date'] = pd.to_datetime(activities_df['start_date_local']).dt.date
        csv_dates = set(df['date'].dt.date)
        new_acts = activities_df[~activities_df['act_date'].isin(csv_dates)]
        if not new_acts.empty:
            st.subheader("🆕 New Activities (not in CSV)")
            st.dataframe(new_acts)
        else:
            st.info("No new activities since last data load.")

# --- SECTION 4: Weekly Summary & Trends ---
st.header(f"🚴‍♂️ Weekly Summary for {username}")
st.dataframe(weekly_summary)

st.subheader("📈 Weekly Training Trends")
fig, ax = plt.subplots()
for metric in ["tss","ctl","atl","tsb"]:
    ax.plot(weekly_summary["week"], weekly_summary[metric], label=metric.upper(), marker='o')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylabel("Score")
ax.set_title("Training Load Metrics")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# --- SECTION 5: GPT-4o Plan Generation ---
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
    client   = OpenAI(api_key=openai_api_key)
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

# --- SECTION 6: Upload Plan to Intervals.icu ---
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
                headers={"Authorization": f"Bearer {st.secrets['intervals_api_key']}",
                         "Content-Type": "application/json"},
                json=payload
            )
            resp.raise_for_status()
            st.success("Plan uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")