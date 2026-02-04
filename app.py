import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FairStay NYC",
    page_icon="🗽",
    layout="wide"
)

# 2. LOAD RESOURCES (Safe Load)
@st.cache_resource
def load_resources():
    required_files = ['train_cleaned.csv', 'airbnb_price_model.pkl', 'le_neighborhood.pkl', 'le_room_type.pkl']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        return None, None, None, None, missing
        
    try:
        df = pd.read_csv('train_cleaned.csv')
        model = joblib.load('airbnb_price_model.pkl')
        le_neigh = joblib.load('le_neighborhood.pkl')
        le_room = joblib.load('le_room_type.pkl')
        return df, model, le_neigh, le_room, []
    except Exception as e:
        return None, None, None, None, [str(e)]

df, model, le_neigh, le_room, missing_files = load_resources()

if missing_files:
    st.error("🚨 CRITICAL ERROR: Files Not Found!")
    st.code("\n".join(missing_files))
    st.stop()

# --- HELPER FUNCTIONS ---
def get_host_type(avail):
    return "🏨 Pro/Business" if avail > 200 else "🏠 Private/Home"

def get_stay_policy(nights):
    if nights < 7: return "Short Stay (<7 days)"
    elif nights < 30: return "Weekly (7-30 days)"
    else: return "Monthly (30+ days)"

df['Host_Type'] = df['availability_365'].apply(get_host_type)
df['Stay_Policy'] = df['minimum_nights'].apply(get_stay_policy)

# 3. SIDEBAR: CONTROLS
st.sidebar.title("FairStay🧳")

# --- DARK MODE FEATURE (NEW) ---
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

if dark_mode:
    # Inject CSS to force Dark Theme colors
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("Predict the **Fair Value** of any listing.")

with st.sidebar.form("prediction_form"):
    st.markdown("### 1. Location & Type")
    selected_neigh = st.selectbox("Neighborhood", le_neigh.classes_)
    selected_room = st.selectbox("Room Type", le_room.classes_)
    
    st.markdown("### 2. The Rules")
    stay_type_input = st.select_slider(
        "Minimum Stay Policy",
        options=["Any (1 night)", "Weekend (2-3 nights)", "Weekly (7+ nights)", "Monthly (30+ nights)"],
        value="Weekend (2-3 nights)"
    )
    
    host_vibe_input = st.radio(
        "Who manages this place?",
        ["🏨 Professional Company", "🏠 Local Resident"],
        help="Professional places are available year-round. Local homes are only available occasionally."
    )
    
    st.markdown("### 3. Reputation")
    num_reviews = st.slider("Number of Reviews", 0, 500, 50)
    
    submit_val = st.form_submit_button("Predict Price 💰")

# 4. PREDICTION LOGIC
if submit_val:
    if "Any" in stay_type_input: min_nights = 1
    elif "Weekend" in stay_type_input: min_nights = 3
    elif "Weekly" in stay_type_input: min_nights = 7
    else: min_nights = 30
        
    if "Professional" in host_vibe_input: availability = 365
    else: availability = 60
    
    neigh_encoded = le_neigh.transform([selected_neigh])[0]
    room_encoded = le_room.transform([selected_room])[0]
    avg_loc = df[df['neighbourhood_group'] == selected_neigh][['latitude', 'longitude']].mean()
    
    input_data = [[neigh_encoded, avg_loc['latitude'], avg_loc['longitude'], room_encoded, min_nights, availability, num_reviews]]
    predicted_price = model.predict(input_data)[0]
    
    st.sidebar.divider()
    st.sidebar.markdown(f"### 🎯 Fair Price: **${predicted_price:.2f}**")
    st.sidebar.caption("Per night (Estimated)")

# 5. MAIN DASHBOARD: THE MAP
st.title("🗽 FairStay NYC: The Map")

# --- FILTERS ROW ---
col1, col2, col3 = st.columns(3)
with col1:
    target_borough = st.selectbox("Filter by Borough", ["All"] + list(df['neighbourhood_group'].unique()))
with col2:
    target_room = st.selectbox("Filter by Room Type", ["All"] + list(df['room_type'].unique()))
with col3:
    max_price = st.slider("Budget Filter ($)", 50, 500, 200)

# Filter Logic
filtered_df = df[df['price'] <= max_price]
if target_borough != "All":
    filtered_df = filtered_df[filtered_df['neighbourhood_group'] == target_borough]
if target_room != "All":
    filtered_df = filtered_df[filtered_df['room_type'] == target_room]

# --- CONTROLS ROW ---
st.divider()
c1, c2 = st.columns([2, 1])

with c1:
    # SMART MAP STYLE LOGIC
    # If Dark Mode is ON, default to "Dark", otherwise "Streets"
    default_index = 2 if dark_mode else 0
    
    map_style_choice = st.radio(
        "Select Map Style:", 
        ["Streets (Google)", "Satellite (Real)", "Dark (Contrast)"], 
        index=default_index,
        horizontal=True
    )

with c2:
    st.markdown(f"**Found {len(filtered_df)} listings**")
    st.caption("🔴**Red** = Pro Business | 🟢 **Green** = Private Home")

# --- MAP RENDERING ---
def get_color(availability):
    return [255, 0, 0, 160] if availability > 200 else [0, 200, 0, 160]
filtered_df['color'] = filtered_df['availability_365'].apply(get_color)

# Style Logic
if "Satellite" in map_style_choice:
    chosen_style = "mapbox://styles/mapbox/satellite-v9"
elif "Dark" in map_style_choice:
    chosen_style = "mapbox://styles/mapbox/dark-v10"
else:
    chosen_style = "mapbox://styles/mapbox/streets-v11"

view_state = pdk.ViewState(
    latitude=filtered_df['latitude'].mean(),
    longitude=filtered_df['longitude'].mean(),
    zoom=11,
    pitch=45
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=100,
    pickable=True,
)

tooltip = {
    "html": "<b>{name}</b><br>Price: ${price}<br>{Host_Type}<br>{Stay_Policy}",
    "style": {"backgroundColor": "black", "color": "white"}
}

st.pydeck_chart(pdk.Deck(
    map_style=chosen_style,
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip,
    api_keys={"mapbox": "pk.eyJ1IjoiYW12aWlpIiwiYSI6ImNtbDgzaWFsYjAzYnEza3ExNnRhenhtbmsifQ.ixmGry-4WWGeit35rX6ThA"}
))

# 6. HIDDEN GEMS TABLE
st.subheader("💎 Hidden Gems (High Quality, Low Price)")
gems = filtered_df[filtered_df['number_of_reviews'] > 50].sort_values(by='price').head(5)

st.dataframe(
    gems[['name', 'price', 'Host_Type', 'Stay_Policy', 'neighbourhood', 'number_of_reviews']],
    hide_index=True,
    use_container_width=True
)