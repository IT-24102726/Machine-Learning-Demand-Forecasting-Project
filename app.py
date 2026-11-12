"""
ChefCast AI — Restaurant Decision Support System
Interactive Streamlit Dashboard for SLIIT IT3091 Machine Learning
Group 2026-AI-36 | Industry Explorer Track

Fully rebuilt to match 9-notebook pipeline architecture.
"""

import os
import sys
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.forecasting import forecast_future_revenue, forecast_future_demand

# ─────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChefCast AI | Restaurant Decision Support",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Header Banner ─────────────────────────────── */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #134e4a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: #fff;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  }
  .hero h1 { font-size: 2rem; font-weight: 800; margin: 0.4rem 0 0.2rem; letter-spacing:-0.5px;}
  .hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }

  /* ── Badge ─────────────────────────────────────── */
  .badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(20,184,166,0.15);
    color: #2dd4bf;
    border: 1px solid rgba(45,212,191,0.35);
    margin-right: 0.4rem;
    margin-bottom: 0.5rem;
  }

  /* ── KPI Card ───────────────────────────────────── */
  .kpi {
    background: #fff;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border-left: 5px solid #0d9488;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    transition: transform .2s, box-shadow .2s;
    height: 100%;
  }
  .kpi:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.1); }
  .kpi-label { color: #64748b; font-size: 0.75rem; font-weight: 700;
               text-transform: uppercase; letter-spacing: 0.6px; }
  .kpi-value { color: #0f172a; font-size: 1.85rem; font-weight: 800; margin: 0.25rem 0; }
  .kpi-sub   { color: #0d9488; font-size: 0.78rem; font-weight: 600; }

  /* ── Section title ──────────────────────────────── */
  .sec-title {
    font-size: 1.15rem; font-weight: 700; color: #1e293b;
    margin: 1.5rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem;
  }

  /* ── Alert / Info boxes ─────────────────────────── */
  .info-box  {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    padding: .9rem 1.1rem; border-radius: 10px;
    color: #166534; font-size: .9rem; margin: .75rem 0;
  }
  .warn-box  {
    background: #fffbeb; border: 1px solid #fde68a;
    padding: .9rem 1.1rem; border-radius: 10px;
    color: #92400e; font-size: .9rem; margin: .75rem 0;
  }
  .note-box  {
    background: #eff6ff; border: 1px solid #bfdbfe;
    padding: .9rem 1.1rem; border-radius: 10px;
    color: #1e40af; font-size: .9rem; margin: .75rem 0;
  }

  /* ── Tab tweaks ─────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-weight: 600;
    font-size: 0.88rem;
  }

  /* ── Table style ────────────────────────────────── */
  .styled-table th { background:#f1f5f9; color:#475569; font-size:.8rem; }

  /* ── Sidebar ────────────────────────────────────── */
  [data-testid="stSidebar"] { background: #0f172a !important; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSlider > div { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    daily_rev = pd.read_csv('data/processed/daily_revenue.csv', parse_dates=['Date'])
    daily_demand_wide = pd.read_csv('data/processed/daily_product_demand.csv', parse_dates=['Date'])
    daily_demand_long = pd.read_csv('data/processed/daily_product_demand_long.csv', parse_dates=['Date'])

    with open('models/metrics.json') as f:
        metrics = json.load(f)
    with open('data/processed/data_summary.json') as f:
        data_summary = json.load(f)

    lens1_preds = pd.read_csv('models/lens1_test_predictions.csv', parse_dates=['Date'])
    lens2_preds = pd.read_csv('models/lens2_test_predictions.csv', parse_dates=['Date'])

    lens1_fi = pd.read_csv('models/lens1_feature_importance.csv')
    lens2_fi = pd.read_csv('models/lens2_feature_importance.csv')

    return (daily_rev, daily_demand_wide, daily_demand_long,
            metrics, data_summary,
            lens1_preds, lens2_preds,
            lens1_fi, lens2_fi)


with st.spinner("🔄 Loading ChefCast AI models and data…"):
    (daily_rev, daily_demand_wide, daily_demand_long,
     metrics, data_summary,
     lens1_preds, lens2_preds,
     lens1_fi, lens2_fi) = load_all_data()

CORE_ITEMS = data_summary['selected_items']
AVG_PRICES = data_summary.get('historical_avg_prices', {})


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <div style="font-size:2.8rem;">🍽️</div>
      <div style="font-size:1.3rem; font-weight:800; color:#2dd4bf; margin-top:0.3rem;">
        ChefCast AI
      </div>
      <div style="font-size:0.75rem; color:#64748b; margin-top:0.15rem;">
        Restaurant Decision Support
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Operational Parameters")

    planning_horizon = st.slider(
        "Forecast Horizon (Days)", min_value=3, max_value=14, value=7,
        help="Number of future days to forecast"
    )

    risk_option = st.select_slider(
        "Kitchen Buffer Strategy",
        options=["Lean (0%)", "Conservative (+10%)", "Balanced (+15%)", "High (+25%)"],
        value="Balanced (+15%)",
        help="Asymmetric safety buffer applied on top of ML forecast to prevent stockouts."
    )
    buffer_map = {"Lean (0%)": 1.00, "Conservative (+10%)": 1.10,
                  "Balanced (+15%)": 1.15, "High (+25%)": 1.25}
    buffer_mult = buffer_map[risk_option]

    st.markdown("---")
    st.markdown(f"### 🧾 Core Dishes ({len(CORE_ITEMS)})")
    for item in CORE_ITEMS:
        price = AVG_PRICES.get(item, 0)
        st.markdown(f"<div style='font-size:0.82rem; padding:0.2rem 0;'>• <b>{item}</b> <span style='color:#64748b'>(≈LKR {price:,.0f})</span></div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    l1_champ = metrics['lens1_revenue'].get('champion_info', {}).get('champion_model', 'Random_Forest')
    l2_champ = metrics['lens2_demand'].get('champion_info', {}).get('champion_model', 'Stacking_Ensemble')
    l1_wape  = metrics['lens1_revenue'].get('champion_info', {}).get('unbiased_test_wape', 0)
    l2_wape  = metrics['lens2_demand'].get('champion_info', {}).get('unbiased_test_wape', 0)

    st.markdown(f"""
    <div style='font-size:0.78rem; line-height:1.8;'>
      <div>📊 <b>L1 Champion:</b> {l1_champ.replace('_',' ')}</div>
      <div style='color:#2dd4bf;'>   Test WAPE: {l1_wape:.1f}%</div>
      <div>🍲 <b>L2 Champion:</b> {l2_champ.replace('_',' ')}</div>
      <div style='color:#2dd4bf;'>   Test WAPE: {l2_wape:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("SLIIT IT3091 | Group 2026-AI-36")
    st.caption("Data: 2024-03-27 → 2026-09-07")


# ─────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <span class="badge">Industry Explorer Track</span>
    <span class="badge">Lens 1: Revenue Planning</span>
    <span class="badge">Lens 2: Kitchen Production</span>
    <span class="badge">11 Core Handmade Dishes</span>
    <span class="badge">127,093 POS Transactions</span>
  </div>
  <h1>🍽️ Restaurant Revenue & Demand Decision Support System</h1>
  <p>Evidence-based ML forecasting translating POS transaction history into weekly revenue planning
  and daily kitchen batch preparation guidance. Champions selected strictly on held-out validation WAPE.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Generate forecasts (cached per horizon/buffer)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_revenue_forecast(horizon, buffer):
    return forecast_future_revenue(horizon=horizon, buffer_pct=buffer)

@st.cache_data(show_spinner=False)
def get_demand_forecast(horizon, buffer):
    return forecast_future_demand(horizon=horizon, buffer_multiplier=buffer)


with st.spinner("⚡ Generating forecasts…"):
    rev_forecast  = get_revenue_forecast(planning_horizon, 0.15)
    dem_forecast  = get_demand_forecast(planning_horizon, buffer_mult)


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Revenue Forecast",
    "🍲 Kitchen Daily Schedule",
    "🏆 Model Leaderboard",
    "📊 Historical Backtesting",
    "🎛️ What-If Simulator",
    "🔍 Data Audit & Methodology",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — LENS 1: REVENUE FORECAST
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">📈 Upcoming Revenue Planning Expectations</div>',
                unsafe_allow_html=True)
    st.markdown("Forward-looking recursive forecast for executive cash-flow planning and supplier settlement schedules.")

    total_rev = rev_forecast['Forecasted Revenue (LKR)'].sum()
    daily_avg = rev_forecast['Forecasted Revenue (LKR)'].mean()
    low_sum   = rev_forecast['Planning Lower Bound'].sum()
    high_sum  = rev_forecast['Planning Upper Bound'].sum()
    q90_buf   = metrics['lens1_revenue'].get('champion_info', {}).get('empirical_residual_q90', 0)

    c1, c2, c3, c4 = st.columns(4)
    kpi_border = ["#0d9488", "#6366f1", "#f59e0b", "#10b981"]
    kpi_data = [
        ("Forecasted Revenue", f"LKR {total_rev:,.0f}", f"Next {planning_horizon} days", kpi_border[0]),
        ("Daily Average", f"LKR {daily_avg:,.0f}", "Per day (forecast)", kpi_border[1]),
        ("Planning Range", f"LKR {low_sum:,.0f}", f"→ {high_sum:,.0f}", kpi_border[2]),
        ("Champion Model", l1_champ.replace('_',' '), f"Test WAPE: {l1_wape:.1f}%", kpi_border[3]),
    ]
    for col, (label, value, sub, border) in zip([c1, c2, c3, c4], kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-left-color:{border};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast Chart ──
    st.markdown(f"#### 📅 {planning_horizon}-Day Forward Forecast  "
                f"({rev_forecast['Date'].min().strftime('%d %b %Y')} → "
                f"{rev_forecast['Date'].max().strftime('%d %b %Y')})")

    fig, ax = plt.subplots(figsize=(14, 4))
    x = rev_forecast['Date']
    ax.fill_between(x,
                    rev_forecast['Planning Lower Bound'],
                    rev_forecast['Planning Upper Bound'],
                    alpha=0.2, color='#0d9488', label='Planning Range (±15%)')
    ax.plot(x, rev_forecast['Forecasted Revenue (LKR)'],
            color='#0d9488', linewidth=2.5, marker='o', markersize=6, label='ML Forecast')

    for _, row in rev_forecast.iterrows():
        ax.annotate(f"LKR {row['Forecasted Revenue (LKR)']:,.0f}",
                    (row['Date'], row['Forecasted Revenue (LKR)']),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, color='#0f172a')

    ax.set_ylabel('Revenue (LKR)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'LKR {v:,.0f}'))
    ax.set_title(f'ChefCast AI — {planning_horizon}-Day Revenue Forecast with Planning Band', fontsize=12)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Forecast Table ──
    st.markdown("##### Detailed Forecast Table")
    display_rev = rev_forecast.copy()
    display_rev['Date'] = display_rev['Date'].dt.strftime('%d %b %Y')
    st.dataframe(
        display_rev.style.format({
            'Forecasted Revenue (LKR)': 'LKR {:,.2f}',
            'Planning Lower Bound'    : 'LKR {:,.2f}',
            'Planning Upper Bound'    : 'LKR {:,.2f}',
        }),
        use_container_width=True,
    )

    st.markdown("""
    <div class="info-box">
    💡 <b>Decision Insight:</b> The ±15% planning band (empirical 90th-percentile residual buffer)
    gives finance a defensible cash-flow corridor for raw-ingredient procurement.
    Peak revenue is concentrated on <b>Fridays and weekends</b> — align staff rosters accordingly.
    </div>""", unsafe_allow_html=True)

    # ── Historical trend ──
    with st.expander("📉 View Historical Revenue Trend (Full Dataset)"):
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        rev_30 = daily_rev.tail(90)
        ax2.plot(rev_30['Date'], rev_30['Revenue'], color='#6366f1', linewidth=1.2, alpha=0.8)
        ax2.fill_between(rev_30['Date'], rev_30['Revenue'], alpha=0.12, color='#6366f1')
        ax2.set_title('Last 90 Days — Actual Daily Revenue')
        ax2.set_ylabel('Revenue (LKR)')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax2.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 2 — LENS 2: KITCHEN DAILY SCHEDULE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">🍲 Kitchen Production Planning & Daily Batch Schedule</div>',
                unsafe_allow_html=True)
    st.markdown("Converts ML demand forecasts into concrete **morning batch preparation targets** for each dish station.")

    sel_item = st.selectbox("Select Dish for Station View:", CORE_ITEMS, index=0)

    item_fc = dem_forecast[dem_forecast['ItemName'] == sel_item].copy()
    total_raw  = item_fc['Forecast_Demand'].sum()
    total_prep = item_fc['Recommended_Kitchen_Prep'].sum()
    avg_price  = AVG_PRICES.get(sel_item, 200.0)
    est_rev    = total_raw * avg_price

    c1, c2, c3 = st.columns(3)
    kpi3 = [
        (f"{planning_horizon}-Day Demand ({sel_item})", f"{total_raw:.0f} portions",
         f"≈ {total_raw/planning_horizon:.1f} portions/day", "#0d9488"),
        (f"Kitchen Prep Target ({risk_option})", f"{total_prep:.0f} portions",
         f"Buffer: +{total_prep-total_raw:.0f} units", "#6366f1"),
        ("Estimated Category Revenue", f"LKR {est_rev:,.0f}",
         f"@ LKR {avg_price:.0f}/portion (VWAP)", "#f59e0b"),
    ]
    for col, (label, value, sub, border) in zip([c1, c2, c3], kpi3):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-left-color:{border};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart per day ──
    st.markdown(f"#### 📊 Daily Batch Schedule — {sel_item}")
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    x3 = np.arange(len(item_fc))
    w = 0.38
    bars1 = ax3.bar(x3 - w/2, item_fc['Forecast_Demand'],   w, color='#0d9488', alpha=0.85, label='ML Forecast')
    bars2 = ax3.bar(x3 + w/2, item_fc['Recommended_Kitchen_Prep'], w, color='#f59e0b', alpha=0.85, label='Kitchen Prep Target')
    ax3.set_xticks(x3)
    ax3.set_xticklabels([f"{r['DayOfWeek'][:3]}\n{r['Date'].strftime('%d %b')}" for _, r in item_fc.iterrows()],
                         fontsize=9)
    ax3.set_ylabel('Portions')
    ax3.set_title(f'{sel_item} — Daily Demand Forecast vs Kitchen Prep Target')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.4)
    for bar in bars1:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("##### Station Daily Schedule Table")
    disp_item = item_fc[['Date', 'DayOfWeek', 'Forecast_Demand', 'Recommended_Kitchen_Prep']].copy()
    disp_item['Date'] = disp_item['Date'].dt.strftime('%d %b %Y')
    disp_item.columns = ['Date', 'Day', 'ML Forecast (portions)', 'Prep Target (portions)']
    st.dataframe(disp_item.style.format({'ML Forecast (portions)': '{:.1f}', 'Prep Target (portions)': '{:.0f}'}),
                 use_container_width=True)

    st.markdown("---")
    # ── Master production sheet ──
    st.markdown("#### 📋 Unified Kitchen Master Production Sheet (All 11 Dishes)")
    master = dem_forecast.pivot(index='ItemName', columns='Date', values='Recommended_Kitchen_Prep')
    master.columns = [d.strftime('%a %d %b') for d in master.columns]

    fig_hm, ax_hm = plt.subplots(figsize=(14, 5))
    sns.heatmap(master, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax_hm,
                linewidths=0.5, cbar_kws={'label': 'Portions'}, annot_kws={'size': 9})
    ax_hm.set_title('Kitchen Prep Targets by Dish × Day', fontsize=12)
    ax_hm.set_xlabel('')
    plt.tight_layout()
    st.pyplot(fig_hm)
    plt.close()

    st.markdown("""
    <div class="info-box">
    💡 <b>Kitchen Ops Decision:</b> Values include your selected safety buffer.
    <b>Darker cells = higher volume days.</b>
    Cross-reference with revenue forecast — high-revenue days should have proportionally higher prep.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">🏆 Model Performance Benchmark</div>', unsafe_allow_html=True)
    st.markdown("""
    **Methodological rigour:** Champion selected on **Validation WAPE only**.
    The Test set (15% of dates) is strictly held-out for unbiased evaluation — never used for selection.
    """)

    def build_leaderboard(metrics_dict, key, champion_name):
        rows = []
        for name, vals in metrics_dict[key].items():
            if name == 'champion_info':
                continue
            v = vals.get('validation', {})
            t = vals.get('test', {})
            is_champ = (name == champion_name)
            rows.append({
                '🏆'          : '🏆' if is_champ else '',
                'Model'       : name.replace('_', ' '),
                'Val WAPE %'  : v.get('WAPE_percent', 0),
                'Val MAE'     : v.get('MAE', 0),
                'Test WAPE %' : t.get('WAPE_percent', 0),
                'Test MAE'    : t.get('MAE', 0),
                'Test RMSE'   : t.get('RMSE', 0),
                'Test R²'     : t.get('R2_Score', 0),
            })
        return pd.DataFrame(rows).sort_values('Val WAPE %').reset_index(drop=True)

    l1_lb = build_leaderboard(metrics, 'lens1_revenue', l1_champ)
    l2_lb = build_leaderboard(metrics, 'lens2_demand',  l2_champ)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Lens 1 — Revenue Models")
        st.dataframe(
            l1_lb.style
            .highlight_min(subset=['Val WAPE %', 'Test WAPE %'], color='#bbf7d0')
            .format({'Val WAPE %': '{:.2f}', 'Val MAE': '{:,.0f}',
                     'Test WAPE %': '{:.2f}', 'Test MAE': '{:,.0f}',
                     'Test RMSE': '{:,.0f}', 'Test R²': '{:.4f}'}),
            use_container_width=True, height=260
        )

    with col_b:
        st.markdown("##### Lens 2 — Demand Models")
        st.dataframe(
            l2_lb.style
            .highlight_min(subset=['Val WAPE %', 'Test WAPE %'], color='#bbf7d0')
            .format({'Val WAPE %': '{:.2f}', 'Val MAE': '{:.2f}',
                     'Test WAPE %': '{:.2f}', 'Test MAE': '{:.2f}',
                     'Test RMSE': '{:.2f}', 'Test R²': '{:.4f}'}),
            use_container_width=True, height=260
        )

    st.markdown("---")

    # ── Feature Importance ──
    st.markdown("##### Feature Importance Analysis (LightGBM)")
    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown("**Lens 1 — Revenue Drivers**")
        fig_fi1, ax_fi1 = plt.subplots(figsize=(7, 5))
        top10_l1 = lens1_fi.head(15)
        ax_fi1.barh(top10_l1['feature'][::-1], top10_l1['importance'][::-1],
                    color='#0d9488', alpha=0.85)
        ax_fi1.set_xlabel('Importance Score')
        ax_fi1.set_title('Revenue: Top 15 Features')
        ax_fi1.grid(axis='x', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_fi1)
        plt.close()

    with fi_col2:
        st.markdown("**Lens 2 — Demand Drivers**")
        fig_fi2, ax_fi2 = plt.subplots(figsize=(7, 5))
        top10_l2 = lens2_fi.head(15)
        ax_fi2.barh(top10_l2['feature'][::-1], top10_l2['importance'][::-1],
                    color='#6366f1', alpha=0.85)
        ax_fi2.set_xlabel('Importance Score')
        ax_fi2.set_title('Demand: Top 15 Features')
        ax_fi2.grid(axis='x', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_fi2)
        plt.close()

    st.markdown("""
    <div class="note-box">
    📌 <b>Interpretation:</b> <code>lag_1</code> dominating confirms strong day-to-day autocorrelation.
    <code>dow_sin/cos</code> signals the weekly trading cycle.
    Item OHE features confirm large demand-level differences between dishes.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — HISTORICAL BACKTESTING
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">📊 Historical Holdout Backtesting</div>', unsafe_allow_html=True)
    st.markdown("Comparing champion vs baseline on the **held-out test split** — data the model never trained on.")

    bt_col1, bt_col2 = st.columns(2)

    with bt_col1:
        st.markdown("##### Lens 1 — Revenue: Champion vs Baseline (Last 60 Days)")
        l1_view = lens1_preds.tail(60).set_index('Date')
        # Flexible column detection
        champ_col_l1 = l1_champ if l1_champ in l1_view.columns else \
                       next((c for c in l1_view.columns if 'Random' in c or 'Stack' in c or 'LightGBM' in c), None)
        naive_col_l1 = next((c for c in l1_view.columns if 'Naive' in c or 'Baseline' in c), None)

        fig_bt1, ax_bt1 = plt.subplots(figsize=(8, 4))
        ax_bt1.plot(l1_view.index, l1_view['Actual'], color='#1e293b', linewidth=2,
                    label='Actual Recorded', zorder=5)
        if champ_col_l1:
            ax_bt1.plot(l1_view.index, l1_view[champ_col_l1], color='#0d9488', linewidth=1.5,
                        linestyle='--', label=f'Champion ({champ_col_l1.replace("_"," ")})', alpha=0.9)
        if naive_col_l1:
            ax_bt1.plot(l1_view.index, l1_view[naive_col_l1], color='#f59e0b', linewidth=1,
                        linestyle=':', label='Seasonal Naive', alpha=0.7)
        ax_bt1.set_ylabel('Revenue (LKR)')
        ax_bt1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
        ax_bt1.legend(fontsize=8)
        ax_bt1.tick_params(axis='x', rotation=25)
        ax_bt1.grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_bt1)
        plt.close()

    with bt_col2:
        st.markdown("##### Lens 2 — Demand: Select Dish to Compare")
        bt_dish = st.selectbox("Dish:", CORE_ITEMS, key='bt_dish')
        l2_view = lens2_preds[lens2_preds['ItemName'] == bt_dish].tail(60).set_index('Date')

        champ_col_l2 = l2_champ if l2_champ in l2_view.columns else \
                       next((c for c in l2_view.columns if 'Random' in c or 'Stack' in c or 'LightGBM' in c), None)
        naive_col_l2 = next((c for c in l2_view.columns if 'Naive' in c or 'Baseline' in c), None)

        fig_bt2, ax_bt2 = plt.subplots(figsize=(8, 4))
        ax_bt2.plot(l2_view.index, l2_view['Actual'], color='#1e293b', linewidth=2,
                    label='Actual Sold', zorder=5)
        if champ_col_l2:
            ax_bt2.plot(l2_view.index, l2_view[champ_col_l2], color='#6366f1', linewidth=1.5,
                        linestyle='--', label=f'Champion ({champ_col_l2.replace("_"," ")})', alpha=0.9)
        if naive_col_l2:
            ax_bt2.plot(l2_view.index, l2_view[naive_col_l2], color='#f59e0b', linewidth=1,
                        linestyle=':', label='Seasonal Naive', alpha=0.7)
        ax_bt2.set_ylabel('Portions Sold')
        ax_bt2.set_title(bt_dish)
        ax_bt2.legend(fontsize=8)
        ax_bt2.tick_params(axis='x', rotation=25)
        ax_bt2.grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_bt2)
        plt.close()

    st.markdown("---")

    # Per-item test WAPE heatmap
    st.markdown("##### Per-Dish Test WAPE Heatmap — All Models (Lens 2)")
    all_model_cols = [c for c in lens2_preds.columns
                      if c not in ['Date', 'ItemName', 'Actual']]
    heatmap_data = {}
    for col in all_model_cols:
        per_item_wape = {}
        for item in CORE_ITEMS:
            sub = lens2_preds[lens2_preds['ItemName'] == item]
            if len(sub) == 0:
                per_item_wape[item] = 0.0
                continue
            yt = sub['Actual'].values
            yp = np.maximum(sub[col].values, 0)
            wape = np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1e-9) * 100
            per_item_wape[item] = round(wape, 1)
        heatmap_data[col.replace('_', ' ')] = per_item_wape

    hm_df = pd.DataFrame(heatmap_data, index=CORE_ITEMS)
    fig_hm2, ax_hm2 = plt.subplots(figsize=(14, 6))
    sns.heatmap(hm_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax_hm2,
                linewidths=0.5, cbar_kws={'label': 'WAPE (%)'}, vmin=0, vmax=100,
                annot_kws={'size': 9})
    ax_hm2.set_title('Per-Dish Test WAPE (%) — All Lens 2 Models (lower = better)', fontsize=12)
    ax_hm2.set_xlabel('Model')
    ax_hm2.set_ylabel('Dish')
    plt.tight_layout()
    st.pyplot(fig_hm2)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 5 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">🎛️ Interactive What-If Scenario Simulator</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Academic Simulation Note:</b> Slider multipliers are user-configurable stress-test assumptions,
    not empirically derived POS econometric parameters.
    </div>""", unsafe_allow_html=True)

    sim1, sim2 = st.columns(2)
    with sim1:
        demand_shock  = st.slider("Simulated Demand Shift (%)", -50, 50, 0, 5)
        price_change  = st.slider("Menu Price Adjustment (%)",  -20, 30, 0, 5)
        holiday_surge = st.checkbox("Long Weekend / Holiday Surge (+15%)")
    with sim2:
        ingredient_cost_pct = st.slider("Ingredient Cost as % of Revenue", 30, 70, 45, 5)
        waste_pct = st.slider("Estimated Spoilage Rate (%)", 2, 15, 8, 1)
        staff_overhead = st.slider("Staff Overhead (LKR/day)", 5000, 30000, 12000, 1000)

    multiplier  = (1 + demand_shock / 100) * (1 + price_change / 100)
    if holiday_surge:
        multiplier *= 1.15

    base_rev  = rev_forecast['Forecasted Revenue (LKR)'].sum()
    sim_rev   = base_rev * multiplier
    rev_delta = sim_rev - base_rev

    ingredient_cost = sim_rev * (ingredient_cost_pct / 100)
    waste_cost = ingredient_cost * (waste_pct / 100)
    total_staff = staff_overhead * planning_horizon
    gross_margin = sim_rev - ingredient_cost - waste_cost - total_staff

    s1, s2, s3, s4 = st.columns(4)
    sim_kpis = [
        ("Base ML Forecast", f"LKR {base_rev:,.0f}", "", "#0d9488"),
        ("Simulated Revenue", f"LKR {sim_rev:,.0f}", f"{rev_delta:+,.0f} ({multiplier*100-100:+.1f}%)", "#6366f1"),
        ("Estimated Waste Cost", f"LKR {waste_cost:,.0f}", f"{waste_pct}% spoilage × {ingredient_cost_pct}% CoGS", "#ef4444"),
        ("Est. Gross Margin", f"LKR {gross_margin:,.0f}", f"After CoGS + waste + staff", "#10b981"),
    ]
    for col, (label, value, sub, border) in zip([s1, s2, s3, s4], sim_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-left-color:{border};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub if sub else '&nbsp;'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_sim, ax_sim = plt.subplots(figsize=(14, 4))
    sim_df = rev_forecast[['Date', 'Forecasted Revenue (LKR)']].copy()
    sim_df['Simulated Scenario'] = sim_df['Forecasted Revenue (LKR)'] * multiplier
    ax_sim.plot(sim_df['Date'], sim_df['Forecasted Revenue (LKR)'],
                color='#0d9488', linewidth=2, marker='o', label='Base ML Forecast')
    ax_sim.plot(sim_df['Date'], sim_df['Simulated Scenario'],
                color='#6366f1', linewidth=2, marker='s', linestyle='--', label='Simulated Scenario')
    ax_sim.fill_between(sim_df['Date'],
                        sim_df['Forecasted Revenue (LKR)'],
                        sim_df['Simulated Scenario'],
                        alpha=0.15, color='#6366f1')
    ax_sim.set_ylabel('Revenue (LKR)')
    ax_sim.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax_sim.set_title('Scenario Stress-Test Comparison')
    ax_sim.legend()
    ax_sim.grid(axis='y', alpha=0.4)
    ax_sim.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    st.pyplot(fig_sim)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 6 — DATA AUDIT & METHODOLOGY
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-title">🔍 Data Quality Audit & Methodology Log</div>',
                unsafe_allow_html=True)
    st.markdown("Full transparency — audit log, data dictionary, feature groups, and architectural decision log.")

    # ── Audit Metrics ──
    st.markdown("#### Raw POS Data Anomaly Audit (NB01 §4)")
    a1, a2, a3, a4, a5 = st.columns(5)
    audit_kpis = [
        ("Raw Transactions", "127,093", "2024–2026", "#0d9488"),
        ("Refunds / Voids", "931 rows", "Qty ≤ 0 filtered", "#ef4444"),
        ("Price Anomalies", "52 rows", "> LKR 20,000", "#f59e0b"),
        ("Pre-2024 Records", "147 rows", "Clock-reset / test rows", "#6366f1"),
        ("Retained (clean)", "125,912 rows", "99.07% retention", "#10b981"),
    ]
    for col, (label, value, sub, border) in zip([a1, a2, a3, a4, a5], audit_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-left-color:{border}; padding:1rem;">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:1.3rem;">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue stats ──
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("#### Lens 1 — Dataset Statistics")
        st.dataframe(pd.DataFrame([{
            'Metric': 'Calendar Days',       'Value': f"{len(daily_rev):,}"},
            {'Metric': 'Mean Daily Revenue',  'Value': f"LKR {daily_rev['Revenue'].mean():,.2f}"},
            {'Metric': 'Median Daily Revenue','Value': f"LKR {daily_rev['Revenue'].median():,.2f}"},
            {'Metric': 'Max Daily Revenue',   'Value': f"LKR {daily_rev['Revenue'].max():,.2f}"},
            {'Metric': 'Zero-Revenue Days',   'Value': f"{(daily_rev['Revenue']==0).sum()}"},
            {'Metric': 'Total Handmade Revenue','Value': f"LKR {daily_rev['Revenue'].sum():,.2f}"},
        ]), use_container_width=True, hide_index=True)

    with col_stat2:
        st.markdown("#### Lens 2 — Product Volume Summary")
        vol_data = []
        for item in CORE_ITEMS:
            if item in daily_demand_wide.columns:
                series = daily_demand_wide[item]
                vol_data.append({
                    'Dish': item,
                    'Total Portions': f"{series.sum():,.0f}",
                    'Mean/Day': f"{series.mean():.1f}",
                    'Max/Day': f"{series.max():.0f}",
                    'Zero Days': f"{(series==0).sum()}",
                })
        st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Data Dictionary ──
    st.markdown("#### Data Dictionary (Raw POS Variables)")
    dict_data = [
        {"Variable": "INVOICENO", "Type": "Categorical/ID", "Meaning": "Unique receipt identifier"},
        {"Variable": "INVDATE",   "Type": "Datetime",       "Meaning": "Transaction timestamp (standardised to ISO)"},
        {"Variable": "ItemID",    "Type": "Integer",        "Meaning": "Internal POS catalogue ID"},
        {"Variable": "ItemName",  "Type": "String",         "Meaning": "Menu item name (uppercase, trimmed)"},
        {"Variable": "UnitPrice", "Type": "Float (LKR)",    "Meaning": "Unit selling price — filter >0 and ≤20,000"},
        {"Variable": "Qty",       "Type": "Float",          "Meaning": "Quantity sold — filter >0 and ≤300"},
        {"Variable": "Cost",      "Type": "Float (LKR)",    "Meaning": "Recorded production/ingredient cost"},
    ]
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Decision Log ──
    st.markdown("#### Formal Architectural Decision Log (NB09 §10)")
    decisions = [
        ("DEC-01", "Track Selection",      "All tracks",
         "Industry Explorer Track",
         "Real restaurant POS with 127k transactions provides authentic ML challenges eligible for bonus marks."),
        ("DEC-02", "Product Selection",     "All 274 items vs 11 core",
         "11 Core Handmade Dishes",
         "Perishable in-house items require active batching decisions; packaged items excluded as irrelevant to kitchen ops."),
        ("DEC-03", "Validation Strategy",   "Random K-Fold vs Temporal",
         "Chronological 70/15/15 Split",
         "Random splitting leaks future demand into past training. Strict temporal split guarantees real-world generalization."),
        ("DEC-04", "Champion Selection",    "Test set vs Validation set",
         "Validation WAPE (strictly held-out test)",
         "Using test set for selection causes optimistic bias. Validation WAPE ensures the test set remains a genuine unbiased benchmark."),
        ("DEC-05", "Feature Engineering",   "Raw dates vs Lag+Rolling",
         "Shifted Lag + Rolling (shift=1)",
         "Raw unlagged features introduce target leakage. Shifting by 1 guarantees zero future-data contamination."),
        ("DEC-06", "Target (Lens 2)",       "Revenue vs Quantity",
         "Daily Quantity (Qty) per dish",
         "Kitchen staff need portion counts, not LKR values. WAPE used instead of MAPE to handle zero-demand days."),
        ("DEC-07", "Zero/Inactive Days",    "Drop vs Zero-fill",
         "Zero-fill with documented caveats",
         "Dropping breaks calendar continuity. Zeros are explicitly flagged as 'no qualifying sales recorded' (closures / downtime)."),
        ("DEC-08", "Ensemble Architecture", "Single model vs Stacking",
         "Stacking with TimeSeriesSplit meta-CV",
         "Stacking combines Ridge's linearity with RF's non-linearity and LightGBM's gradient boosting — robust to different error types."),
    ]
    dec_df = pd.DataFrame(decisions,
                          columns=['ID', 'Area', 'Options Considered', 'Decision Made', 'Rationale'])
    st.dataframe(dec_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.82rem; padding:0.5rem 0 1rem;">
  <b>ChefCast AI</b> — IT3091 Machine Learning Group Assignment | SLIIT Faculty of Computing | Group 2026-AI-36<br>
  <span style="color:#64748b;">
    IT24102572 · IT24102664 · IT24102726 · IT24102913 &nbsp;|&nbsp;
    Streamlit · Scikit-Learn · LightGBM · XGBoost · Pandas · Matplotlib · Seaborn
  </span>
</div>
""", unsafe_allow_html=True)
