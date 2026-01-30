import streamlit as st
import matplotlib.pyplot as plt
import os
import pandas as pd

# ── Private data upload ── (only shown when deployed online)
st.sidebar.title("Upload Confidential Data (one-time)")

uploaded_main = st.sidebar.file_uploader(
    "1. Fencing preseason Final Data.csv",
    type="csv"
)
uploaded_iso = st.sidebar.file_uploader(
    "2. Fencing_Preseason_Isokinetic data.csv",
    type="csv"
)

if uploaded_main and uploaded_iso:
    # Load once and store in session state
    if "df_main" not in st.session_state:
        st.session_state.df_main = pd.read_csv(uploaded_main, encoding='latin-1')
        st.session_state.df_iso = pd.read_csv(uploaded_iso, encoding='latin-1')
        
        # Apply your column cleaning
        st.session_state.df_main.columns = st.session_state.df_main.columns.str.replace('\xa0', ' ', regex=False).str.strip()
        st.session_state.df_iso.columns = st.session_state.df_iso.columns.str.replace('\xa0', ' ', regex=False).str.strip()
        
        st.sidebar.success("Files uploaded successfully! Dashboard ready.")
else:
    st.sidebar.warning("Please upload both files above to continue.")
    st.stop()

# Use session state DataFrames everywhere
df = st.session_state.df_main
iso_df = st.session_state.df_iso

# Import all your plotting functions
from data_transform import (
    plot_jump_height, plot_peak_power_bw,
    plot_sl_board_jump, plot_knee_to_wall,
    plot_slj_asymmetry, plot_slh_rsi,
    plot_isokinetic_torque, plot_isokinetic_3box,
    plot_ybalance_directional, plot_cmj_vs_team
)

# === PAGE CONFIG & STYLE ===
st.set_page_config(page_title="Fencing Preseason Dashboard", layout="wide", page_icon="fencing")

st.markdown("""
<style>
    .header-title {
        background-color: #001f3f; color: white; padding: 20px; text-align: center;
        border-radius: 12px; font-size: 32px; font-weight: bold; margin: 20px 0;
    }
    .sub-header {
        background-color: #008080; color: white; padding: 15px; text-align: center;
        border-radius: 10px; font-size: 20px; font-weight: bold; margin: 15px 0;
    }
    .plot-box {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1); margin: 20px 0; text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stApp { background-color: #f5f7fa; }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR FILTERS ===
st.sidebar.header("Filters")

# Athlete filter — now uses uploaded df
athlete_list = ['All Athletes'] + sorted(df['Athletes'].dropna().unique().tolist())
selected_athlete = st.sidebar.selectbox("Athlete", athlete_list)

# Group filter
group_list = ['All Groups'] + sorted(df['Group'].dropna().unique().tolist())
selected_group = st.sidebar.selectbox("Group", group_list)

# Dominant Side filter - ONLY Left and Right
side_options = ['All Sides', 'Left', 'Right']
selected_side = st.sidebar.selectbox("Dominant Side", side_options)

# === APPLY FILTERS ===
filtered_df = df.copy()
if selected_athlete != 'All Athletes':
    filtered_df = filtered_df[filtered_df['Athletes'] == selected_athlete]
if selected_group != 'All Groups':
    filtered_df = filtered_df[filtered_df['Group'] == selected_group]
if selected_side != 'All Sides':
    filtered_df = filtered_df[filtered_df['Dominant side'] == selected_side[0]]  # 'L' or 'R'

if filtered_df.empty:
    st.error("No athletes match the selected filters.")
    st.stop()

# Use first athlete for display
row = filtered_df.iloc[0]
selected_athlete = row['Athletes']
gender = "Male" if "M" in row.get('Group', '') else "Female"
dom = row.get('Dominant side', 'R')

# === MAIN TITLE ===
st.markdown(f'<div class="header-title">ATHLETE PROFILE: {selected_athlete} • {gender} • Dominant: {dom}</div>',
            unsafe_allow_html=True)

# === ROW 1 ===
st.markdown('<div class="header-title">PERFORMANCE DATA (CMJ)</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="sub-header">JUMP HEIGHT (CM)</div>', unsafe_allow_html=True)
    fig = plot_jump_height(filtered_df, selected_athlete)
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    

with col2:
    st.markdown('<div class="sub-header">PEAK POWER / BW</div>', unsafe_allow_html=True)
    fig = plot_peak_power_bw(filtered_df, selected_athlete)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

# === ROW 2 ===
st.markdown("""
<div style="display: flex; gap: 20px; margin: 40px 0;">
    <div style="flex: 1; background-color: #001f3f; color: white; padding: 20px; text-align: center; border-radius: 15px; font-size: 28px; font-weight: bold;">
        SL BOARD JUMP (CM)
    </div>
    <div style="flex: 1; background-color: #001f3f; color: white; padding: 20px; text-align: center; border-radius: 15px; font-size: 28px; font-weight: bold;">
        KNEE-TO-WALL (CM)
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig = plot_sl_board_jump(filtered_df, selected_athlete)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No SL Board Jump data")

with col2:
    fig = plot_knee_to_wall(filtered_df, selected_athlete)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No Knee-to-Wall data")

# === ROW 3 ===
st.markdown('<div class="header-title">ASYMMETRY</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="sub-header">SL JUMP</div>', unsafe_allow_html=True)
    fig = plot_slj_asymmetry(filtered_df, selected_athlete)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

with col2:
    st.markdown('<div class="sub-header">SLH JUMP</div>', unsafe_allow_html=True)
    fig = plot_slh_rsi(filtered_df, selected_athlete)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

# === ROW 4 ===
st.markdown("""
<div style="background-color: #001f3f; color: white; padding: 22px; text-align: center; border-radius: 15px; font-size: 30px; font-weight: bold; margin: 40px 0;">
    KNEE ISOKINETIC TEST
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig = plot_isokinetic_torque(iso_df, selected_athlete)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No torque data available")

with col2:
    fig = plot_isokinetic_3box(iso_df, selected_athlete)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No isokinetic profile data")

# === ROW 5 ===
st.markdown("""
<div style="display: flex; gap: 20px; margin: 40px 0;">
    <div style="flex: 1; background-color: #001f3f; color: white; padding: 22px; text-align: center; border-radius: 15px; font-size: 28px; font-weight: bold;">
        Y-BALANCE TEST
    </div>
    <div style="flex: 1; background-color: #001f3f; color: white; padding: 22px; text-align: center; border-radius: 15px; font-size: 28px; font-weight: bold;">
        CMJ MEAN VALUE VS TEAM AVERAGE
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig = plot_ybalance_directional(filtered_df, selected_athlete)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Y-Balance Test data")

with col2:
    fig = plot_cmj_vs_team(df, selected_athlete)  # ← df is now from session state
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CMJ comparison data")


