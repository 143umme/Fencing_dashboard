# data_transform.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

def load_data():
    """
    Load both CSV files from the 'data' folder (same level as this file).
    Works perfectly with your current folder structure.
    """
    # Automatically find the 'data' folder next to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
    
    main_file = os.path.join(data_folder, "Fencing preseason Final Data.csv")
    iso_file = os.path.join(data_folder, "Fencing_Preseason_Isokinetic data.csv")
    
    # Friendly error if files are missing
    if not os.path.exists(main_file):
        raise FileNotFoundError(
            f"File not found:\n{main_file}\n\n"
            "Please make sure 'Fencing preseason Final Data.csv' is in the 'data' folder."
        )
    if not os.path.exists(iso_file):
        raise FileNotFoundError(
            f"File not found:\n{iso_file}\n\n"
            "Please make sure 'Fencing_Preseason_Isokinetic data.csv' is in the 'data' folder."
        )
    
    # Load with correct encoding
    df_main = pd.read_csv(main_file, encoding='latin-1')
    df_iso = pd.read_csv(iso_file, encoding='latin-1')
    
    # Clean column names
    df_main.columns = df_main.columns.str.strip()
    df_iso.columns = df_iso.columns.str.strip()
    
    return df_main, df_iso


def clean_data(df_main, df_isokinetic):
    """
    Clean both datasets - fix weird spaces and convert numbers
    """
    
    def clean_df(df, id_cols):
        df.columns = df.columns.str.replace('\xa0', ' ', regex=False).str.strip()
        for col in df.columns:
            if col not in id_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.copy()
    
    df_main = clean_df(df_main, ['Athletes', 'Group', 'Dominant side'])
    df_isokinetic = clean_df(df_isokinetic, ['Athlete', 'Gender', 'Dominant Side'])
    
    return df_main, df_isokinetic


# Test it (optional - remove later)
if __name__ == "__main__":
    df, iso = load_data()
    df, iso = clean_data(df, iso)
    print("SUCCESS! Both files loaded and cleaned.")
    print(f"Main data: {df.shape}, Athletes: {len(df['Athletes'].unique())}")
    print(f"Isokinetic data: {iso.shape}")


# Jump height
def plot_jump_height(df, athlete_name):
    """
    Creates the exact Jump Height (CM) graph from your Jupyter notebook.
    Works dynamically for any selected athlete.
    """
    # Filter athlete
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    
    athlete = row.iloc[0]
    mean_sep = pd.to_numeric(athlete.get('CMJ MEAN Jump Height (Imp-Mom) (cm)'), errors='coerce')
    
    if pd.isna(mean_sep):
        return None  # No data
    
    years = [2025, 2026, 2027]
    mean_values = [mean_sep, np.nan, np.nan]
    
    fig, ax = plt.subplots(figsize=(45, 40))
    
    # Main line + points
    ax.plot(years, mean_values,
            marker='o', markersize=60, linewidth=3.5,
            color='#003366', label='MEAN Jump Height', zorder=3)
    
    # Highlight 2025
    ax.plot(2025, mean_sep, 'o', markersize=100, color='#003366', zorder=7)
    ax.text(2025, mean_sep - 2, f"{mean_sep:.1f}", ha='center', va='top',
            fontweight='bold', color='#003366', fontsize=60)
    
    # "No Data" labels
    for y in [2026, 2027]:
        ax.text(y, 30, "No Data", ha='center', va='center', fontsize=45,
                color='gray', style='italic', alpha=0.7)
    
    # Formatting (exactly like your code)
    ax.set_xlim(2024.5, 2027.5)
    ax.set_ylim(0, 80)
    ax.tick_params(axis='y', labelsize=60, width=3, length=8)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=60, fontweight='bold')
    ax.set_ylabel('Jump Height (cm)', fontsize=60, fontweight='bold')
    ax.grid(True, axis='y', linestyle='-', color='lightgray', alpha=0.7, linewidth=0.8)
    ax.grid(True, axis='x', linestyle='-', color='lightgray', alpha=0.3)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92),
              ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=60)
    
    # Clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

# CMJ Peak Power
def plot_peak_power_bw(df, athlete_name):
    """
    Creates the exact Peak Power / BW bar chart from your Jupyter notebook.
    Works dynamically for any selected athlete.
    """
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    
    athlete = row.iloc[0]
    col = 'CMJ MEAN Peak Power / BM (W/kg)'
    value_2025 = pd.to_numeric(athlete.get(col), errors='coerce')
    
    if pd.isna(value_2025):
        return None
    
    years = [2025, 2026, 2027]
    values = [value_2025, np.nan, np.nan]
    
    fig, ax = plt.subplots(figsize=(45, 40))
    
    # Bars
    ax.bar(years, values,
           color=['#4da6ff', '#cccccc', '#cccccc'],
           width=0.6, edgecolor='black', linewidth=1.2, zorder=3)
    
    # Annotate 2025 value
    ax.text(2025, value_2025 + 1, f"{value_2025:.1f}",
            ha='center', va='bottom', fontweight='bold',
            color='#4da6ff', fontsize=60)
    
    # "No Data" labels
    for year in [2026, 2027]:
        ax.text(year, 20, "No Data", ha='center', va='center',
                fontsize=45, color='gray', style='italic', alpha=0.8)
    
    # Formatting - exactly like your code
    ax.set_xlim(2024.2, 2027.8)
    ax.set_ylim(0, max(60, value_2025 * 1.3))
    ax.tick_params(axis='y', labelsize=60, width=3, length=8)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=60, fontweight='bold' )
    ax.set_ylabel('Power (W/kg)', fontsize=60, fontweight='bold')
    
    ax.grid(True, axis='y', linestyle='-', color='lightgray', alpha=0.7, linewidth=0.8)
    ax.grid(True, axis='x', linestyle='-', color='lightgray', alpha=0.3)
    
    # Legend
    legend_elements = [Patch(facecolor='#4da6ff', edgecolor='black', label='2025 (September)')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92),
              ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=60)
    
    # Clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

# single leg board jump
def plot_sl_board_jump(df, athlete_name):
    """
    Creates the exact SL Board Jump (CM) graph.
    Fixed for pandas Series issue.
    """
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    
    athlete = row.iloc[0]
    
    nd = pd.to_numeric(athlete.get('SLBJ MAX Non Dominant (cm)'), errors='coerce')
    d = pd.to_numeric(athlete.get('SLBJ MAX Dominant (cm)'), errors='coerce')
    
    if pd.isna(nd) and pd.isna(d):
        return None
    
    target_years = [2025, 2026, 2027]
    nd_values = [nd, np.nan, np.nan]
    d_values = [d, np.nan, np.nan]
    
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Convert to numpy arrays to avoid pandas indexing issues
    years_arr = np.array(target_years)
    nd_arr = np.array(nd_values)
    d_arr = np.array(d_values)
    
    # Non-Dominant (Green)
    mask_nd = ~np.isnan(nd_arr)
    if mask_nd.any():
        ax.plot(years_arr, nd_arr, color='#2E8B57', linewidth=5, marker='o',
                markersize=30, markerfacecolor='#90EE90', markeredgewidth=3, markeredgecolor='black',
                label='Non-Dominant', zorder=5)
        ax.scatter(years_arr[mask_nd], nd_arr[mask_nd],
                   s=1800, color='#2E8B57', edgecolor='black', linewidth=3, zorder=6)
    
    # Dominant (Orange)
    mask_d = ~np.isnan(d_arr)
    if mask_d.any():
        ax.plot(years_arr, d_arr, color='#FF8C00', linewidth=5, marker='o',
                markersize=30, markerfacecolor='#FFD580', markeredgewidth=3, markeredgecolor='black',
                label='Dominant', zorder=5)
        ax.scatter(years_arr[mask_d], d_arr[mask_d],
                   s=1800, color='#FF8C00', edgecolor='black', linewidth=3, zorder=6)
    
    # "No Data" labels
    for i, year in enumerate(target_years):
        if np.isnan(nd_values[i]) and np.isnan(d_values[i]):
            ax.text(year, 160, "No Data", ha='center', va='center', fontsize=25,
                    fontweight='bold', color='gray',
                    bbox=dict(boxstyle="round,pad=0.7", facecolor='white', edgecolor='lightgray', linewidth=2))
    
    # Styling
    ax.set_xlim(2024.5, 2027.5)
    ax.set_xticks(target_years)
    ax.set_xticklabels(['2025', '2026', '2027'], fontsize=30, fontweight='bold')
    ax.set_ylabel("Jump Distance (cm)", fontsize=30, fontweight='bold', labelpad=15)
    ax.set_ylim(50, 230)
    ax.set_yticks(np.arange(50, 231, 20))
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.legend(fontsize=25, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

# Knee to wall

def plot_knee_to_wall(df, athlete_name):
    """
    Knee-to-Wall (KTW) — FINAL BOSS VERSION
    Exactly how you want it: clean, bold, beautiful
    """
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None

    athlete = row.iloc[0]

    non_dom = pd.to_numeric(athlete.get('KTW Non Dominant (cm)'), errors='coerce')
    dom = pd.to_numeric(athlete.get('KTW Dominant (cm)'), errors='coerce')

    if pd.isna(non_dom) and pd.isna(dom):
        return None

    years = [2025, 2026, 2027]
    non_dom_vals = [non_dom, np.nan, np.nan]
    dom_vals = [dom, np.nan, np.nan]

    fig, ax = plt.subplots(figsize=(20, 18))

    x = np.arange(len(years))

    # Bars — thicker edges
    ax.bar(x - 0.2, non_dom_vals, width=0.4, label='Non-Dominant', color='skyblue', edgecolor='black', linewidth=2)
    ax.bar(x + 0.2, dom_vals,     width=0.4, label='Dominant',     color='navy',     edgecolor='black', linewidth=2)

    # Normative line — green dashed, NO BOX
    ax.axhline(10, color='green', linestyle='--', linewidth=3, alpha=0.8)

    # Value labels — BIG & BOLD
    if not pd.isna(non_dom):
        ax.text(0 - 0.2, non_dom + 0.8, f"{non_dom:.1f}", ha='center', va='bottom',
                fontsize=30, fontweight='bold', color='black')
    if not pd.isna(dom):
        ax.text(0 + 0.2, dom + 0.8, f"{dom:.1f}", ha='center', va='bottom',
                fontsize=30, fontweight='bold', color='white')

    # X & Y labels — size 15, bold
    
    ax.set_ylabel("KTW Distance (cm)", fontsize=30, fontweight='bold', color='#001f3f', labelpad=15)

    # Tick labels — size 15, bold
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=30, width=2, length=8)
    ax.tick_params(axis='x', labelsize=30, width=2, length=8)

    ax.set_ylim(0, 20)

    # Legend
    ax.legend(fontsize=25, loc='upper right', frameon=True, fancybox=True, shadow=False)

    # === YOUR REQUESTED CLEAN LOOK (ADDED!) ===
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.set_facecolor('#f8f9fa')        # Light gray background
    fig.patch.set_facecolor('white')   # White figure background

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', color='lightgray')

    plt.tight_layout()
    return fig

# Single leg jump
def plot_slj_asymmetry(df, athlete_name):
    df = df.copy()
    df.columns = df.columns.str.replace('\xa0', ' ', regex=False).str.strip()
    print("Cleaned columns:", df.columns.tolist())

    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    athlete = row.iloc[0]

    # EXACT COLUMN NAMES FROM YOUR EXCEL FILE
    metrics = [
        ('SLJ MEAN Non Dominant Landing impulse (N/s)', 'SLJ MEAN Dominant Landing impulse (N/s)', 
         'SLJ MEAN Non Dominant/Dominant Landing impulse ASYM %', 'Landing impulse (N/s)'),
        ('SLJ MEAN Non Dominant Concentric impulse 100ms   (N/s)', 'SLJ MEAN Dominant Concentric impulse 100ms  (N/s)', 
         'SLJ MEAN Non Dominant/Dominant Concentric impulse 100ms  ASYM %', 'Concentric impulse 100ms (N/s)'),
        ('SLJ MEAN Non Dominant EDRFD/BM   (N/s/kg)', 'SLJ MEAN Dominant EDRFD/BM  (N/s/kg)', 
         'SLJ MEAN Non Dominant/Dominant EDRFD/BM  ASYM %', 'EDRFD / BM (N/s/kg)'),
        ('SLJ MEAN Non Dominant Peak Power / BM  (W/kg)', 'SLJ MEAN Dominant Peak Power / BM  (W/kg)', 
         'SLJ MEAN Non Dominant/Dominant Peak Power / BM  ASYM %', 'Peak Power / BM (W/kg)'),
        ('SLJ MEAN Non Dominant Jump Height (Imp-Mom)  (cm)', 'SLJ MEAN Dominant Jump Height (Imp-Mom)  (cm)', 
         'SLJ MEAN Non Dominant/Dominant Jump Height (Imp-Mom)  ASYM %', 'Jump Height (cm)'),  # ← TWO SPACES HERE
    ]

    # RSI — EXACT FROM YOUR DATA
    rsi_non_dom = 'SLJ MEAN Non Dominant RSI-modified (Imp-Mom) (m/s)'
    rsi_dom = 'SLJ MEAN Dominant RSI-modified (Imp-Mom) (m/s)'
    rsi_asym = 'SLJ MEAN Non Dominant/Dominant RSI-modified (Imp-Mom) ASYM %'

    main_data = []
    labels = []

    for nd_col, dom_col, asym_col, label in metrics:
        nd = pd.to_numeric(athlete.get(nd_col), errors='coerce')
        dom = pd.to_numeric(athlete.get(dom_col), errors='coerce')
        asym_raw = athlete.get(asym_col)
        
        # DEBUG: Print what we found
        # print(f"Looking for: {asym_col}")
        # print(f"Found: {asym_raw}")

        asym = pd.to_numeric(asym_raw, errors='coerce')

        if pd.isna(nd) or pd.isna(dom):
            continue

        main_data.append((nd, dom, asym))
        labels.append(label)

    # RSI
    rsi_nd = pd.to_numeric(athlete.get(rsi_non_dom), errors='coerce')
    rsi_d = pd.to_numeric(athlete.get(rsi_dom), errors='coerce')
    rsi_a = pd.to_numeric(athlete.get(rsi_asym), errors='coerce')

    if not main_data and pd.isna(rsi_nd) and pd.isna(rsi_d):
        return None

    fig = plt.figure(figsize=(60, 56))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    for i, (nd, dom, asym) in enumerate(main_data):
        y = len(main_data) - 1 - i
        ax1.plot([nd, dom], [y, y], color='gray', lw=3)
        ax1.scatter(nd, y, s=4800, color='#2e8b57', edgecolor='black', lw=3)
        ax1.scatter(dom, y, s=4800, color='#ff8c00', edgecolor='black', lw=3)
        
        if pd.notna(asym) and asym != 0:
            ax1.text((nd + dom)/2, y + 0.3, f"{asym:.1f}%", ha='center', va='bottom',
                     fontsize=80, fontweight='bold', color='green')

    ax1.set_yticks(range(len(main_data)))
    ax1.set_yticklabels(labels[::-1], fontsize=80, fontweight='bold')
    ax1.set_xlim(0, 160)
    ax1.set_xticks(np.arange(0, 161, 20))
    ax1.tick_params(axis='x', labelsize=85)
    ax1.grid(True, axis='x', alpha=0.4, linestyle='--')

    ax2 = fig.add_subplot(gs[1])
    if pd.notna(rsi_nd) and pd.notna(rsi_d):
        ax2.plot([rsi_nd, rsi_d], [0.5, 0.5], color='gray', lw=3)
        ax2.scatter(rsi_nd, 0.5, s=4800, color='#2e8b57', edgecolor='black', lw=3)
        ax2.scatter(rsi_d, 0.5, s=4800, color='#ff8c00', edgecolor='black', lw=3)
        if pd.notna(rsi_a) and rsi_a != 0:
            ax2.text((rsi_nd + rsi_d)/2, 0.52, f"{rsi_a:.1f}%", ha='center', va='bottom',
                     fontsize=80, fontweight='bold', color='green')

    ax2.set_yticks([0.5])
    ax2.set_yticklabels(['RSI-modified (Imp-Mom) (m/s)'], fontsize=80, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.set_xticks(np.arange(0, 1.01, 0.1))
    ax2.tick_params(axis='x', labelsize=85)

    legend_elements = [
        Patch(facecolor='#2e8b57', edgecolor='black', label='Non-Dominant'),
        Patch(facecolor='#ff8c00', edgecolor='black', label='Dominant')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.98),
               fontsize=80, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig
# Single leg hoop jump
def plot_slh_rsi(df, athlete_name):
    """
    Creates the exact SLH Mean RSI graph (Row 3, Right).
    Works perfectly for any selected athlete.
    """
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    
    athlete = row.iloc[0]
    name = athlete.get('Athletes', 'Unknown Athlete').strip()
    
    # Column names - exactly as in your data
    L_COL = 'SLH MEAN Non Dominant RSI (m/s)'
    R_COL = 'SLH MEAN Dominant RSI (m/s)'
    ASYM_COL = 'SLH MEAN Non Dominant/Dominant RSI ASYM %'
    
    # Parse values
    def parse_asym(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.endswith(('R', 'L')):
            s = s[:-1].strip()
        s = s.replace('%', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    
    L = pd.to_numeric(athlete.get(L_COL), errors='coerce')
    R = pd.to_numeric(athlete.get(R_COL), errors='coerce')
    asym = parse_asym(athlete.get(ASYM_COL))
    
    # If no RSI data
    if pd.isna(L) and pd.isna(R):
        return None
    
    fig, ax = plt.subplots(figsize=(32, 22))
    y = 0
    
    # Line between points
    if not pd.isna(L) and not pd.isna(R):
        ax.plot([L, R], [y, y], color='gray', linewidth=3, zorder=1)
    
    # Non-Dominant (blue)
    if not pd.isna(L):
        ax.scatter(L, y, s=2800, color='#1f77b4', edgecolor='black', linewidth=1.5,
                   zorder=2, label='Non-Dominant')
    
    # Dominant (red)
    if not pd.isna(R):
        ax.scatter(R, y, s=2800, color='#d62728', edgecolor='black', linewidth=1.5,
                   zorder=2, label='Dominant')
    
    # Asymmetry % inside the line
    if not pd.isna(L) and not pd.isna(R) and not pd.isna(asym):
        mid = (L + R) / 2
        ax.text(mid, y, f"{asym:.1f}%", ha='center', va='center',
                fontsize=40, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='darkred', edgecolor='none'))
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=45)
    
    # X-axis
    ax.set_xlim(0.1, 0.6)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontsize=45)
    ax.set_xlabel("RSI (m/s)", fontsize=45)
    
    # Y-axis: Athlete name
    ax.set_yticks([y])
   
    ax.tick_params(axis='y', length=0)
    
    # Title
    ax.set_title("SLH MEAN RSI", fontsize=40, fontweight='bold', pad=12)
    
    # Clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('lightgray')
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.7, zorder=0)
    
    plt.tight_layout()
    return fig

#Isokinetic Peak torque
def plot_isokinetic_torque(df_iso, athlete_name):
    # Clean columns and names
    df_iso = df_iso.copy()
    df_iso.columns = df_iso.columns.str.replace('\xa0', ' ', regex=False).str.strip()
    df_iso['Athlete'] = df_iso['Athlete'].str.strip()

    # Find athlete (flexible matching)
    row = df_iso[df_iso['Athlete'].str.contains(athlete_name.strip(), case=False, na=False)]
    if row.empty:
        return None
    athlete = row.iloc[0]
    name = athlete['Athlete']

    # BULLETPROOF GENDER DETECTION (exactly like your working code)
    gender_raw = str(athlete.get('Gender', '')).strip()
    if gender_raw.lower().startswith('m') or gender_raw in ['M', '1']:
        norm_extensor = 240
        norm_flexor = 140
        gender_str = "Male"
    else:
        norm_extensor = 155
        norm_flexor = 95
        gender_str = "Female"

    # Extract values — EXACT column names from your CSV
    dom_ext = pd.to_numeric(athlete.get('Dominant_Extensor(Nm)'), errors='coerce')
    non_ext = pd.to_numeric(athlete.get('Non Dominant_Extensor(Nm)'), errors='coerce')
    dom_flx = pd.to_numeric(athlete.get('Dominant_Flexor(Nm)'), errors='coerce')
    non_flx = pd.to_numeric(athlete.get('Non Dominant_Flexor(Nm)'), errors='coerce')

    # Check if ANY data exists
    if pd.isna(dom_ext) and pd.isna(non_ext) and pd.isna(dom_flx) and pd.isna(non_flx):
        return None

    fig, ax = plt.subplots(figsize=(18, 16))

    # Plot Dominant Leg
    if pd.notna(dom_flx) and pd.notna(dom_ext):
        ax.scatter(dom_flx, dom_ext, color='#1f77b4', s=800, edgecolors='black', linewidth=2, zorder=5,
                   label='Dominant Leg')
        ax.annotate(f'Dom:({dom_flx:.0f}, {dom_ext:.0f})',
                    (dom_flx + 6, dom_ext + 8), color='#1f77b4',
                    fontsize=16, fontweight='bold')

    # Plot Non-Dominant Leg
    if pd.notna(non_flx) and pd.notna(non_ext):
        ax.scatter(non_flx, non_ext, color='#d62728', s=800, edgecolors='black', linewidth=2, zorder=5,
                   label='Non-Dominant Leg')
        ax.annotate(f'Non-Dom:({non_flx:.0f}, {non_ext:.0f})',
                    (non_flx + 6, non_ext + 8), color='#d62728',
                    fontsize=16, fontweight='bold')

    # Normative lines
    ax.axvline(norm_flexor, color='green', linestyle='--', linewidth=3, alpha=0.8,
               label=f'Norm Flexor ({norm_flexor} Nm)')
    ax.axhline(norm_extensor, color='green', linestyle='--', linewidth=3, alpha=0.8,
               label=f'Norm Extensor ({norm_extensor} Nm)')

    # Title and labels
    ax.set_title(f'Peak Torque at 60°/s\n{name} • {gender_str}',
                 fontsize=22, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Flexor Peak Torque (Nm)', fontsize=23, fontweight='bold')
    ax.set_ylabel('Extensor Peak Torque (Nm)', fontsize=23, fontweight='bold')

    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(40, 250)
    ax.set_ylim(40, 350)
    ax.tick_params(axis='x', labelsize=23)  # Bigger numbers on X
    ax.tick_params(axis='y', labelsize=23)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=20)

    plt.tight_layout()
    return fig
# FINAL Isokinetic 3-Box Profile 
def plot_isokinetic_3box(df_iso, athlete_name):
    """
    3-Box Isokinetic Profile — EXACTLY like your original notebook.
    Works perfectly for ALL athletes in the dashboard.
    """
    # Clean column names
    df_iso = df_iso.copy()
    df_iso.columns = df_iso.columns.str.replace('\xa0', ' ', regex=False).str.strip()
    df_iso['Athlete'] = df_iso['Athlete'].str.strip()

    # Find athlete (flexible matching)
    row = df_iso[df_iso['Athlete'].str.contains(athlete_name.strip(), case=False, na=False)]
    if row.empty:
        return None
    ath = row.iloc[0]
    name = ath['Athlete']

    # GENDER DETECTION
    gender_raw = str(ath.get('Gender', '')).strip().upper()
    gender = "MALE" if gender_raw in ['M', 'MALE', '男'] else "FEMALE"

    # Gender-specific norms
    quad_low = 2.5 if gender == "MALE" else 2.1
    hs_low   = 1.5 if gender == "MALE" else 1.2

    # SAFE PARSING (exactly like your original)
    def parse_range(val):
        if pd.isna(val): 
            return np.nan
        s = str(val).strip()
        if '-' in s:
            try:
                a, b = s.split('-')
                return round((float(a) + float(b)) / 2, 2)
            except:
                return np.nan
        try:
            return round(float(s), 2)
        except:
            return np.nan

    def parse_float(val):
        if pd.isna(val): 
            return np.nan
        try:
            return round(float(str(val).replace('%','').strip()), 1)
        except:
            return np.nan

    # Extract values
    quad = parse_range(ath.get('Normalized_Extensor(Nm/Kg)'))
    hs   = parse_range(ath.get('Normalized_Flexor(Nm/Kg)'))
    hq_dom = parse_float(ath.get('Dominant_H/Q Ratio(%)'))
    hq_ndom = parse_float(ath.get('Non Dominant_H/Q Ratio(%)'))
    ext_asym = abs(parse_float(ath.get('Ext_Asym_%')) or 0)
    flex_asym = abs(parse_float(ath.get('Flex_Asym_%')) or 0)

    # Color logic
    quad_color = 'lightcoral' if pd.isna(quad) or quad < quad_low else 'lightgreen'
    hs_color   = 'lightcoral' if pd.isna(hs) or hs < hs_low else 'lightgreen'
    hq_dom_color = 'lightcoral' if pd.isna(hq_dom) or hq_dom < 60 else 'lightgreen'
    hq_ndom_color = 'lightcoral' if pd.isna(hq_ndom) or hq_ndom < 60 else 'lightgreen'
    ext_asym_color = 'lightcoral' if ext_asym > 15 else 'lightgreen'
    flex_asym_color = 'lightcoral' if flex_asym > 15 else 'lightgreen'

    # === PLOT — EXACTLY YOUR ORIGINAL 3-BOX ===
    fig = plt.figure(figsize=(40, 36))
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle(f"{name} • {gender}", fontsize=60, fontweight='bold', y=0.98, color='#2c3e50')

    # Box 1: Peak Torque / BW
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 8); ax1.axis('off')
    ax1.text(5, 7.5, "Peak Torque / BW (Nm/kg)", ha='center', fontsize=50, fontweight='bold')
    for x, y in [(1,4.5), (5.5,4.5), (1,1.5), (5.5,1.5)]:
        ax1.add_patch(Rectangle((x,y), 4, 2.5, fill=True, facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax1.text(3, 5.75, "Extensor", ha='center', fontsize=50, fontweight='bold')
    ax1.text(7.5, 5.75, "Non-Dom", ha='center', fontsize=50)
    ax1.text(3, 2.75, "Flexor", ha='center', fontsize=52, fontweight='bold')
    ax1.text(7.5, 2.75, "Dom", ha='center', fontsize=50)
    if not pd.isna(quad):
        ax1.text(7.5, 5.75, f"{quad:.2f}", ha='center', va='center', fontsize=55, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=1", facecolor=quad_color, edgecolor='black'))
    if not pd.isna(hs):
        ax1.text(7.5, 2.75, f"{hs:.2f}", ha='center', va='center', fontsize=55, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=1", facecolor=hs_color, edgecolor='black'))

    # Box 2: H/Q Ratio
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis('off')
    ax2.text(5, 7.5, "H/Q Ratio (%)", ha='center', fontsize=50, fontweight='bold')
    ax2.add_patch(Rectangle((1, 4.5), 8, 2.5, fill=True, facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax2.add_patch(Rectangle((1, 1.5), 8, 2.5, fill=True, facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax2.text(3, 5.75, "Non-Dom", ha='center', fontsize=52, fontweight='bold')
    ax2.text(3, 2.75, "Dom", ha='center', fontsize=55, fontweight='bold')
    if not pd.isna(hq_ndom):
        ax2.text(7, 5.75, f"{hq_ndom:.1f}", ha='center', va='center', fontsize=55, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=1", facecolor=hq_ndom_color, edgecolor='black'))
    if not pd.isna(hq_dom):
        ax2.text(7, 2.75, f"{hq_dom:.1f}", ha='center', va='center', fontsize=55, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=1", facecolor=hq_dom_color, edgecolor='black'))

    # Box 3: Leg Asymmetry
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 8); ax3.axis('off')
    ax3.text(5, 7.5, "Leg Asymmetry (%)", ha='center', fontsize=50, fontweight='bold')
    ax3.add_patch(Rectangle((1, 4.5), 8, 2.5, fill=True, facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax3.add_patch(Rectangle((1, 1.5), 8, 2.5, fill=True, facecolor='#e8e8e8', edgecolor='black', linewidth=2))
    ax3.text(3, 5.75, "D/ND Extensor", ha='center', fontsize=46, fontweight='bold')
    ax3.text(3, 2.75, "D/ND Flexor", ha='center', fontsize=50, fontweight='bold')
    ax3.text(7, 5.75, f"{ext_asym:.1f}%", ha='center', va='center', fontsize=55, fontweight='bold',
             bbox=dict(boxstyle="round,pad=1", facecolor=ext_asym_color, edgecolor='black'))
    ax3.text(7, 2.75, f"{flex_asym:.1f}%", ha='center', va='center', fontsize=55, fontweight='bold',
             bbox=dict(boxstyle="round,pad=1", facecolor=flex_asym_color, edgecolor='black'))

    plt.tight_layout()
    return fig

# Y-Balance Test 

def plot_ybalance_directional(df, athlete_name):
    """
    Creates the exact Y-Balance Test directional graph.
    Shows Left vs Right leg, 94% norm line, <4% diff norm, and inter-limb differences.
    Works for any athlete from the main dataset.
    """
    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    
    athlete_data = row.iloc[0]
    
    # Define metrics and columns
    metrics = ['Anterior', 'Medial', 'Lateral', 'Composite']
    left_cols = [
        'YBT LA Relative (normalized) (%)',
        'YBT LM Relative (normalized) (%)',
        'YBT LL Relative (normalized) (%)',
        'L_Composite (%)'
    ]
    right_cols = [
        'YBT RA Relative (normalized) (%)',
        'YBT RM Relative (normalized) (%)',
        'YBT RL Relative (normalized) (%)',
        'R_Composite (%)'
    ]
    
    left_leg_values = [pd.to_numeric(athlete_data.get(col), errors='coerce') for col in left_cols]
    right_leg_values = [pd.to_numeric(athlete_data.get(col), errors='coerce') for col in right_cols]
    
    # Check if any data exists
    if all(pd.isna(v) for v in left_leg_values + right_leg_values):
        return None
    
    fig = go.Figure()
    
    # Left Leg (Blue)
    fig.add_trace(go.Scatter(
        y=metrics,
        x=left_leg_values,
        mode='lines+markers',
        name='Left Leg',
        line=dict(color='blue'),
        marker=dict(size=15)
    ))
    
    # Right Leg (Orange)
    fig.add_trace(go.Scatter(
        y=metrics,
        x=right_leg_values,
        mode='lines+markers',
        name='Right Leg',
        line=dict(color='orange'),
        marker=dict(size=15)
    ))
    
    # Normative line at 94%
    fig.add_vline(x=94, line_dash="dash", line_color="red",
                  annotation_text="Normative Value: 94%", annotation_position="top left")
    
    # Top-right annotation: Diff <4%
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        text="Diff Normative Value: <4%",
        showarrow=False,
        font=dict(size=13),
        bgcolor="white",
        opacity=True,
        xanchor="right", yanchor="top"
    )
    
    # Add inter-limb difference % between lines
    for i, metric in enumerate(metrics):
        L, R = left_leg_values[i], right_leg_values[i]
        if not pd.isna(L) and not pd.isna(R):
            diff = abs(L - R)
            mid_x = (L + R) / 2
            fig.add_annotation(
                y=metric,
                x=mid_x,
                text=f"{diff:.1f}%",
                showarrow=False,
                font=dict(size=15),
                bgcolor="white",
                opacity=0.8
            )
    
    # Layout
    fig.update_layout(
        title='',
        yaxis_title='Test Metric',
        xaxis_title='Reach Percentage (%)',
        height=800,  # MUCH BIGGER
        width=1200,
        font=dict(family="Arial Black", size=13, color="black"),
        showlegend=True,
        legend=dict(x=1.0, y=0.9, xanchor="right", yanchor="top"),
        template='plotly_white',
        margin=dict(t=40, b=60, l=80, r=80)
    )
    
    # Dynamic x-axis range
    all_vals = [v for v in left_leg_values + right_leg_values if not pd.isna(v)]
    if all_vals:
        x_min = min(all_vals) - 10
        x_max = max(all_vals) + 10
        fig.update_xaxes(range=[max(0, x_min), x_max])
    
    return fig
    
def plot_cmj_vs_team(df, athlete_name):
    # Clean columns (your ultimate cleaning)
    df = df.copy()
    df.columns = (df.columns
                  .str.replace('\xa0', ' ', regex=False)
                  .str.replace('ï¿½', '', regex=True)
                  .str.replace('�', '', regex=True)
                  .str.replace(r'[^\x00-\x7F]+', '', regex=True)
                  .str.strip()
                  .str.replace(' +', ' ', regex=True))

    row = df[df['Athletes'] == athlete_name]
    if row.empty:
        return None
    athlete_data = row.iloc[0]

    # Get athlete gender from 'Group' column
    athlete_gender_raw = athlete_data.get('Group', '').strip()
    if 'M' in athlete_gender_raw.upper():
        athlete_gender = 'Male'
    elif 'F' in athlete_gender_raw.upper():
        athlete_gender = 'Female'
    else:
        athlete_gender = 'Unknown'

    # FIXED: NaN-safe gender-specific team average
    def safe_mean_gender(col, gender):
        if gender == 'Unknown':
            return np.nan

        # Fill NaN → empty string so .contains() never returns NaN
        mask = df['Group'].fillna('').str.upper().str.contains(gender[0])

        # Only keep rows where mask is True (exclude NaN mask rows)
        gender_df = df[mask & mask.notna()]

        if gender_df.empty:
            return np.nan

        return gender_df[col].replace([np.inf, -np.inf], np.nan).dropna().mean()

    # === MAIN METRICS ===
    main_metrics = [
        ("Jump Height (Imp-Mom)(cm)", 'CMJ MEAN Jump Height (Imp-Mom) (cm)'),
        ("Peak Power / BM (W/Kg)", 'CMJ MEAN Peak Power / BM (W/kg)'),
        ("EDRFD/BM", 'CMJ MEAN EDRFD/BM'),
        ("Concentric impulse 100ms", 'CMJ MEAN Concentric impulse 100ms'),
        ("Landing impulse", 'CMJ MEAN Landing impulse')
    ]

    main_athlete_vals = []
    main_team_vals = []
    main_labels = []

    for label, col in main_metrics:
        val = pd.to_numeric(athlete_data.get(col), errors='coerce')
        if pd.isna(val):
            continue
        val = float(val)

        avg = safe_mean_gender(col, athlete_gender)
        if pd.isna(avg):
            continue

        main_athlete_vals.append(val)
        main_team_vals.append(float(avg))
        main_labels.append(label)

    # === RSI ===
    rsi_label = "RSI-modified (Imp-Mom)"
    rsi_col = 'CMJ MEAN RSI-modified (Imp-Mom)'
    rsi_athlete = pd.to_numeric(athlete_data.get(rsi_col), errors='coerce')
    rsi_team = safe_mean_gender(rsi_col, athlete_gender)
    rsi_ok = pd.notna(rsi_athlete) and pd.notna(rsi_team)

    if not main_labels and not rsi_ok:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
        subplot_titles=("", "")
    )

    # === MAIN METRICS ===
    if main_labels:
        fig.add_trace(
            go.Scatter(
                x=main_athlete_vals,
                y=main_labels,
                mode='markers',
                name=athlete_name,
                marker=dict(color='#2ca02c', size=[abs(v) * 0.5 for v in main_athlete_vals], sizemode='area', sizemin=20,
                            line=dict(width=1, color='black')),
                hovertemplate='<b>%{y}</b><br>Athlete: %{x:.1f}<extra></extra>',
                legendgroup="athlete"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=main_team_vals,
                y=main_labels,
                mode='markers',
                name=f'{athlete_gender} Team Average',
                marker=dict(color='#ff7f0e', size=[abs(v) * 0.5 for v in main_team_vals], sizemode='area', sizemin=20,
                            line=dict(width=1, color='black')),
                hovertemplate='<b>%{y}</b><br>Team Avg: %{x:.1f}<extra></extra>',
                legendgroup="team"
            ),
            row=1, col=1
        )

    # === RSI ===
    if rsi_ok:
        fig.add_trace(
            go.Scatter(
                x=[rsi_athlete],
                y=[rsi_label],
                mode='markers',
                name=athlete_name,
                showlegend=False,
                marker=dict(color='#2ca02c', size=rsi_athlete * 70, sizemode='area', sizemin=20,
                            line=dict(width=1, color='black')),
                hovertemplate='<b>%{y}</b><br>Athlete: %{x:.3f}<extra></extra>',
                legendgroup="athlete"
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[rsi_team],
                y=[rsi_label],
                mode='markers',
                name=f'{athlete_gender} Team Average',
                showlegend=False,
                marker=dict(color='#ff7f0e', size=rsi_team * 80, sizemode='area', sizemin=20,
                            line=dict(width=1, color='black')),
                hovertemplate='<b>%{y}</b><br>Team Avg: %{x:.3f}<extra></extra>',
                legendgroup="team"
            ),
            row=2, col=1
        )

    # === LAYOUT ===
    fig.update_layout(
        height=800,
        width=1200,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
        margin=dict(l=180, r=80, t=100, b=60)
    )

    # Row 1
    fig.update_yaxes(title_text="Metric", row=1, col=1, categoryorder='array', categoryarray=main_labels)
    if main_athlete_vals:
        max_main = max(main_athlete_vals + main_team_vals)
        fig.update_xaxes(title_text="", range=[0, max_main * 1.15], row=1, col=1)

    # Row 2 (RSI)
    if rsi_ok:
        fig.update_xaxes(range=[0, 1], tick0=0, dtick=0.1, row=2, col=1)
        fig.update_yaxes(tickvals=[0], ticktext=[rsi_label], row=2, col=1)

    return fig