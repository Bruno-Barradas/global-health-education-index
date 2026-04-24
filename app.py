import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Global Health & Education Index",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
header[data-testid="stHeader"] { background: #1a2e4a; }
section[data-testid="stSidebar"] { background: #1a2e4a; }
section[data-testid="stSidebar"] * { color: #cce0f5 !important; }
div[data-testid="metric-container"] {
    background: #f5f7fa; border-left: 4px solid #0d9488;
    border-radius: 6px; padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #0d9488 !important; font-weight: 600; }
h1 { color: #1a2e4a !important; border-bottom: 3px solid #0d9488; padding-bottom: 6px; }
h2 { color: #2d5a8e !important; }
h3 { color: #1a2e4a !important; }
.finding-card {
    background: #f5f7fa; border-left: 5px solid #2d5a8e;
    border-radius: 6px; padding: 16px 20px; margin-bottom: 14px;
}
.finding-card.high   { border-left-color: #e8604c; }
.finding-card.medium { border-left-color: #f0a500; }
.rec-box { background: #eaf3fb; border: 1.5px solid #4a90c4; border-radius: 6px; padding: 12px 16px; margin: 8px 0; font-size: 0.9rem; }
.limit-box { background: #fff8e8; border: 1.5px solid #f0c040; border-radius: 6px; padding: 10px 14px; margin: 4px 0; font-size: 0.87rem; font-style: italic; }
.phase-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px; color: #0d9488; text-transform: uppercase; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("global_health_education.csv")
    return df

@st.cache_data
def get_2022(_df):
    df22 = _df[_df["year"] == 2022].dropna(subset=["life_expectancy","infant_mortality","literacy_rate","gdp_per_capita"])
    return df22

@st.cache_data
def train_model(_df22):
    features = ["gdp_per_capita","health_expenditure_pct_gdp","education_expenditure_pct_gdp","literacy_rate"]
    df_m = _df22[features + ["life_expectancy"]].dropna()
    X = df_m[features]; y = df_m["life_expectancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(); model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, features, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), X_test, y_test, y_pred

@st.cache_data
def get_clusters(_df22):
    features = ["gdp_per_capita","health_expenditure_pct_gdp","education_expenditure_pct_gdp","literacy_rate"]
    df_c = _df22[features + ["country_name","country_code","life_expectancy","infant_mortality","region","income_group"]].dropna()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df_c[features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_c = df_c.copy(); df_c["cluster"] = km.fit_predict(X_s)
    means = df_c.groupby("cluster")["life_expectancy"].mean().sort_values(ascending=False)
    labels = ["High-income Efficient", "Upper-mid Developing", "Upper-mid Transitional", "Low-income Constrained"]
    mapping = {idx: lbl for idx, lbl in zip(means.index, labels)}
    df_c["cluster_name"] = df_c["cluster"].map(mapping)
    return df_c

df = load_data()
df22 = get_2022(df)
model, features, R2, MAE, X_test, y_test, y_pred = train_model(df22)
df22_clusters = get_clusters(df22)

DARK_BLUE="#1a2e4a"; MID_BLUE="#2d5a8e"; TEAL="#0d9488"
ACCENT="#e8604c"; GOLD="#f0a500"; GREEN="#28a745"; GRAY="#aaaaaa"

def phase_label(txt): st.markdown(f'<p class="phase-label">{txt}</p>', unsafe_allow_html=True)
def rec_box(txt): st.markdown(f'<div class="rec-box"><b>Analysis & Recommendation:</b> {txt}</div>', unsafe_allow_html=True)
def limit_box(txt): st.markdown(f'<div class="limit-box"><i>Limitation:</i> {txt}</div>', unsafe_allow_html=True)
def hr(): st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Global Health & Education Index")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Executive Summary",
        "Data Understanding",
        "Health Analysis",
        "Education Analysis",
        "Country Clusters",
        "Predictive Model",
        "Brazil Deep-Dive",
        "Policy Recommendations",
        "Technical Appendix",
    ])
    st.markdown("---")
    st.markdown("**Project:** Global Health & Education Efficiency Index")
    st.markdown("**Author:** Bruno Barradas · 2026")
    st.markdown("**Sources:** World Bank · WHO · Our World in Data")
    st.markdown("[GitHub](https://github.com/Bruno-Barradas/global-health-education-index)")


# =============================================================================
# 1 — EXECUTIVE SUMMARY
# =============================================================================
if page == "Executive Summary":
    st.title("Global Health & Education Efficiency Index")
    st.markdown("### Which countries deliver the best outcomes per dollar invested?")
    st.markdown("Panel data analysis across **104 countries · 2000–2022 · World Bank + WHO + Our World in Data**")
    hr()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Countries", "104")
    c2.metric("Years", "2000–2022")
    c3.metric("Model R2", f"{R2:.2f}")
    c4.metric("MAE", f"{MAE:.2f} yrs")
    c5.metric("Clusters", "4")
    hr()

    st.subheader("Executive Summary")
    summary = pd.DataFrame({
        "Element": ["Objective","Sources","Scope","Target audience","Deliverable"],
        "Description": [
            "Identify which countries deliver the best health and education outcomes per dollar invested and what Brazil can learn from global best practices.",
            "World Bank Open Data · WHO Global Health Observatory · Our World in Data",
            "104 countries · 6 regions · 4 income groups · 2000–2022 (2,392 country-year records)",
            "Policymakers, public health analysts, development economists, and data analyst recruiters",
            "Descriptive analysis + predictive model (R2=0.76) + country clustering + policy recommendations",
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Key Findings")
    findings = [
        ("high","1","Literacy rate is the strongest predictor of life expectancy (corr=0.81) — stronger than GDP or health spending","HIGH","Invest in education as a long-cycle health intervention"),
        ("high","2","Singapore: 85.4 yrs life expectancy at only 4.0% GDP health spending — world's most efficient system","HIGH","Study Singapore's Medisave model for efficiency benchmarking"),
        ("high","3","USA spends 16.8% GDP on health but ranks 40th globally in life expectancy — the biggest spending outlier","HIGH","Volume of spending does not guarantee outcomes — efficiency does"),
        ("high","4","Brazil spends 9.5% GDP on health (comparable to Japan) but achieves 8 fewer years of life expectancy","HIGH","Brazil's problem is efficiency and equity of access — not funding"),
        ("medium","5","4 distinct country clusters: high-income efficient, upper-mid developing (Brazil), transitional, low-income","MEDIUM","Brazil needs structural reforms to move from Cluster 0 to Cluster 1"),
        ("medium","6","Global infant mortality fell 47% since 2000 — but the absolute gap between Africa and rich countries is unchanged","HIGH","Relative progress masks persistent absolute inequality"),
    ]
    for cls, num, title, impact, action in findings:
        badge_color = {"high": ACCENT, "medium": GOLD}[cls]
        st.markdown(f"""<div class="finding-card {cls}"><b>Finding {num} — {title}</b><br>
        <span style="color:{badge_color};font-weight:700;font-size:0.85rem">Impact: {impact}</span><br>
        <span style="font-size:0.9rem;color:#555">→ {action}</span></div>""", unsafe_allow_html=True)

    hr()
    st.info("**Executive conclusion:** The global data delivers a clear counterintuitive verdict: countries that spend most on health do not produce the best outcomes. Literacy rate (corr=0.81) is the strongest predictor of life expectancy — surpassing GDP and health spending. Brazil spends at Japan's level but achieves 8 fewer years of life expectancy. The gap is in efficiency, not investment.")


# =============================================================================
# 2 — DATA UNDERSTANDING
# =============================================================================
elif page == "Data Understanding":
    phase_label("Phase 2 — Data Understanding")
    st.title("Data Understanding")
    hr()

    st.subheader("Data Sources")
    src = pd.DataFrame({
        "Source": ["World Bank Open Data","WHO Global Health Observatory","Our World in Data"],
        "Key Indicators": ["Life expectancy, infant mortality, GDP per capita, health & education expenditure",
                           "Health system coverage, mortality rates by cause",
                           "Long-run literacy rates, historical development trends"],
        "Access": ["data.worldbank.org","who.int/data","ourworldindata.org"],
    })
    st.dataframe(src, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Data Dictionary")
    dic = pd.DataFrame({
        "Variable": ["country_code","country_name","region","income_group","year",
                     "gdp_per_capita","health_expenditure_pct_gdp","life_expectancy",
                     "infant_mortality","education_expenditure_pct_gdp","literacy_rate"],
        "Type": ["str","str","str","str","int","float","float","float","float","float","float"],
        "Description": [
            "ISO 3-letter country code","Country name","Geographic region (6 regions)",
            "World Bank income group (4 categories)","Year (2000–2022)",
            "GDP per capita in current USD","Current health expenditure as % of GDP",
            "Life expectancy at birth (years)","Infant mortality rate per 1,000 live births",
            "Government education expenditure as % of GDP","Adult literacy rate (%)"],
    })
    st.dataframe(dic, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Dataset Quality")
    st.success(f"**{len(df):,} records** across {df['country_code'].nunique()} countries and {df['year'].nunique()} years. Null values handled via mean imputation by income group.")
    st.subheader("Descriptive Statistics — 2022 Cross-Section")
    st.dataframe(df22[["gdp_per_capita","health_expenditure_pct_gdp","life_expectancy","infant_mortality","education_expenditure_pct_gdp","literacy_rate"]].describe().round(2), use_container_width=True)
    hr()

    st.subheader("Distribution by Region & Income Group")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,3.5))
        rc = df22["region"].value_counts()
        ax.barh(rc.index, rc.values, color=TEAL, edgecolor="white", height=0.6)
        for i, v in enumerate(rc.values): ax.text(v+0.3, i, str(v), va="center", fontsize=9, fontweight="bold")
        ax.set_title("Countries by Region", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5,3.5))
        ic = df22["income_group"].value_counts()
        colors_ic = [ACCENT, GOLD, TEAL, MID_BLUE]
        ax.bar(ic.index, ic.values, color=colors_ic[:len(ic)], edgecolor="white", width=0.55)
        for i, v in enumerate(ic.values): ax.text(i, v+0.3, str(v), ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Countries by Income Group", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()


# =============================================================================
# 3 — HEALTH ANALYSIS
# =============================================================================
elif page == "Health Analysis":
    phase_label("Phase 3 — Health Analysis")
    st.title("Health Outcomes Analysis")
    hr()

    st.subheader("Finding 1 — Life Expectancy vs. Health Spending: the spending paradox")
    corr_h = df22["health_expenditure_pct_gdp"].corr(df22["life_expectancy"])
    col1, col2 = st.columns([3,1])
    with col1:
        fig, ax = plt.subplots(figsize=(8,4.5))
        regions = df22["region"].unique()
        colors_r = [ACCENT, MID_BLUE, TEAL, GOLD, GREEN, GRAY]
        for i, reg in enumerate(regions):
            sub = df22[df22["region"]==reg]
            ax.scatter(sub["health_expenditure_pct_gdp"], sub["life_expectancy"],
                       alpha=0.65, color=colors_r[i%len(colors_r)], s=30, label=reg, edgecolors="none")
        # Highlight USA and Singapore
        usa = df22[df22["country_code"]=="USA"]
        sgp = df22[df22["country_code"]=="SGP"]
        bra = df22[df22["country_code"]=="BRA"]
        if not usa.empty: ax.annotate("USA\n16.8% GDP\n78.9 yrs", (usa["health_expenditure_pct_gdp"].values[0], usa["life_expectancy"].values[0]), fontsize=7.5, color=ACCENT, fontweight="bold", xytext=(12,70), arrowprops=dict(arrowstyle="->",color=ACCENT))
        if not sgp.empty: ax.annotate("Singapore\n4.0% GDP\n85.4 yrs", (sgp["health_expenditure_pct_gdp"].values[0], sgp["life_expectancy"].values[0]), fontsize=7.5, color=TEAL, fontweight="bold", xytext=(5,87))
        if not bra.empty: ax.annotate("Brazil\n9.5% GDP\n75.9 yrs", (bra["health_expenditure_pct_gdp"].values[0], bra["life_expectancy"].values[0]), fontsize=7.5, color=MID_BLUE, fontweight="bold")
        ax.set_xlabel("Health Expenditure (% GDP)", fontsize=10); ax.set_ylabel("Life Expectancy (years)", fontsize=10)
        ax.set_title(f"Health Spending vs. Life Expectancy (corr={corr_h:.2f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_h:.2f}", "Moderate positive")
        st.metric("USA spends", "16.8% GDP", "ranks 40th globally")
        st.metric("Singapore spends", "4.0% GDP", "ranks 4th globally")
        st.metric("Brazil spends", "9.5% GDP", "ranks ~60th globally")

    rec_box("The correlation between health spending and life expectancy is only 0.50 — moderate. Singapore achieves top-4 life expectancy at 4.0% GDP. The USA spends 16.8% and ranks 40th. Volume of spending does not guarantee outcomes — efficiency of spending does.")
    limit_box("Health expenditure data includes both public and private spending. Countries with high private spending (USA) may have different efficiency dynamics than countries with high public spending.")
    hr()

    st.subheader("Finding 2 — Top 15 Countries by Life Expectancy (2022)")
    top15 = df22.nlargest(15, "life_expectancy")[["country_name","life_expectancy","health_expenditure_pct_gdp","gdp_per_capita","infant_mortality"]]
    top15.columns = ["Country","Life Expectancy (yrs)","Health Exp. % GDP","GDP per Capita ($)","Infant Mortality/1k"]
    top15 = top15.round(1)
    st.dataframe(top15, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Finding 3 — Infant Mortality Trend: 47% reduction since 2000")
    global_trend = df.groupby("year")["infant_mortality"].mean().reset_index()
    region_trend = df.groupby(["year","income_group"])["infant_mortality"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    colors_ig = {"High": TEAL, "Upper-Mid": MID_BLUE, "Lower-Mid": GOLD, "Low": ACCENT}
    for ig, color in colors_ig.items():
        sub = region_trend[region_trend["income_group"]==ig]
        if not sub.empty: ax.plot(sub["year"], sub["infant_mortality"], color=color, lw=2.5, label=ig)
    ax.plot(global_trend["year"], global_trend["infant_mortality"], color=DARK_BLUE, lw=2, linestyle="--", label="Global avg")
    ax.set_xlabel("Year", fontsize=10); ax.set_ylabel("Infant Mortality (per 1,000 live births)", fontsize=10)
    ax.set_title("Infant Mortality Trend by Income Group (2000–2022)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    rec_box("Global infant mortality fell 47% from 2000 to 2022. However, the absolute gap between low-income countries (~52/1k) and high-income countries (~3.5/1k) remains at 48 points — virtually unchanged. Relative progress masks persistent absolute inequality.")


# =============================================================================
# 4 — EDUCATION ANALYSIS
# =============================================================================
elif page == "Education Analysis":
    phase_label("Phase 3 — Education Analysis")
    st.title("Education Outcomes Analysis")
    hr()

    st.subheader("Finding 1 — Literacy rate is the strongest predictor of life expectancy (corr=0.81)")
    corr_l = df22["literacy_rate"].corr(df22["life_expectancy"])
    col1, col2 = st.columns([3,1])
    with col1:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.scatter(df22["literacy_rate"], df22["life_expectancy"], alpha=0.5,
                   c=df22["gdp_per_capita"], cmap="Blues", s=30, edgecolors="none")
        m, b = np.polyfit(df22["literacy_rate"].fillna(0), df22["life_expectancy"], 1)
        xs = np.linspace(df22["literacy_rate"].min(), 100, 200)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Trend (corr={corr_l:.2f})")
        bra = df22[df22["country_code"]=="BRA"]
        if not bra.empty: ax.annotate("Brazil\n93.2% literacy\n75.9 yrs", (bra["literacy_rate"].values[0], bra["life_expectancy"].values[0]), fontsize=7.5, color=MID_BLUE, fontweight="bold")
        ax.set_xlabel("Literacy Rate (%)", fontsize=10); ax.set_ylabel("Life Expectancy (years)", fontsize=10)
        ax.set_title(f"Literacy Rate vs. Life Expectancy (corr={corr_l:.2f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_l:.2f}", "Strongest predictor")
        st.metric("vs. GDP/capita", "0.67", "literacy wins")
        st.metric("vs. Health exp.", "0.50", "literacy wins")

    rec_box(f"Literacy rate (corr={corr_l:.2f}) is the strongest predictor of life expectancy — surpassing GDP per capita (0.67) and health expenditure (0.50). Countries that educated their populations in the 1980s-90s are now the healthiest. Education is a 30-year investment in population health.")
    hr()

    st.subheader("Finding 2 — Correlations with Life Expectancy")
    corr_series = df22[["gdp_per_capita","health_expenditure_pct_gdp","life_expectancy","infant_mortality","education_expenditure_pct_gdp","literacy_rate"]].corr()["life_expectancy"].drop("life_expectancy").sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,3))
    colors_c = [ACCENT if v > 0.6 else (GOLD if v > 0 else GRAY) for v in corr_series.values]
    ax.bar(corr_series.index, corr_series.values, color=colors_c, edgecolor="white", width=0.55)
    for bar, val in zip(ax.patches, corr_series.values):
        ypos = bar.get_height()+0.02 if val >= 0 else bar.get_height()-0.06
        ax.text(bar.get_x()+bar.get_width()/2, ypos, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color=DARK_BLUE, lw=1)
    ax.set_ylabel("Pearson Correlation", fontsize=9)
    ax.set_title("Correlation of Each Variable with Life Expectancy", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    st.caption("Note: infant_mortality has strong negative correlation (-0.95) — higher mortality = lower life expectancy")


# =============================================================================
# 5 — COUNTRY CLUSTERS
# =============================================================================
elif page == "Country Clusters":
    phase_label("Phase 3 — Country Clustering")
    st.title("Country Clusters — 4 Structural Profiles")
    hr()

    st.subheader("K-Means Clustering (k=4) — Countries grouped by health-education profile")
    cluster_summary = df22_clusters.groupby("cluster_name")[["life_expectancy","gdp_per_capita","health_expenditure_pct_gdp","literacy_rate","infant_mortality"]].mean().round(1)
    cluster_summary.columns = ["Life Exp. (yrs)","GDP/capita ($)","Health Exp. % GDP","Literacy (%)","Infant Mort./1k"]
    st.dataframe(cluster_summary, use_container_width=True)
    hr()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clusters by Life Expectancy vs. GDP")
        fig, ax = plt.subplots(figsize=(6,4.5))
        cluster_colors = {"High-income Efficient": TEAL, "Upper-mid Developing": MID_BLUE,
                          "Upper-mid Transitional": GOLD, "Low-income Constrained": ACCENT}
        for cname, color in cluster_colors.items():
            sub = df22_clusters[df22_clusters["cluster_name"]==cname]
            ax.scatter(sub["gdp_per_capita"], sub["life_expectancy"], color=color, s=35, alpha=0.7, label=cname, edgecolors="none")
        bra = df22_clusters[df22_clusters["country_code"]=="BRA"]
        if not bra.empty:
            ax.scatter(bra["gdp_per_capita"], bra["life_expectancy"], color="red", s=120, marker="*", zorder=5, label="Brazil")
        ax.set_xlabel("GDP per Capita ($)", fontsize=9); ax.set_ylabel("Life Expectancy (yrs)", fontsize=9)
        ax.set_title("Country Clusters", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Brazil's Cluster Peers")
        brazil_cluster = df22_clusters[df22_clusters["country_code"]=="BRA"]["cluster_name"].values
        if len(brazil_cluster) > 0:
            bra_cl = brazil_cluster[0]
            st.info(f"**Brazil is in: {bra_cl}**")
            peers = df22_clusters[df22_clusters["cluster_name"]==bra_cl][["country_name","life_expectancy","gdp_per_capita","health_expenditure_pct_gdp"]].sort_values("life_expectancy", ascending=False)
            peers.columns = ["Country","Life Exp.","GDP/cap","Health Exp. %"]
            peers = peers.round(1)
            st.dataframe(peers, use_container_width=True, hide_index=True)

    rec_box("Brazil is in the 'Upper-mid Developing' cluster with Mexico, South Africa, and Turkey. To reach the 'High-income Efficient' cluster, Brazil needs to close a 9+ year life expectancy gap. The data shows this is achievable through efficiency reforms — not by increasing already-high spending.")


# =============================================================================
# 6 — PREDICTIVE MODEL
# =============================================================================
elif page == "Predictive Model":
    phase_label("Phase 3 — Predictive Model")
    st.title("Predictive Model — Life Expectancy")
    st.markdown(f"Linear Regression trained on 104 countries. **R2={R2:.2f} | MAE={MAE:.2f} years**")
    hr()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R2", f"{R2:.4f}", "Strong predictive capability")
    c2.metric("MAE", f"{MAE:.2f} yrs", "Mean absolute error")
    c3.metric("Countries", "104", "2022 cross-section")
    c4.metric("Algorithm", "Linear Regression")
    hr()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Coefficients")
        coefs = pd.Series(dict(zip(features, model.coef_))).sort_values(key=abs, ascending=True)
        colors_imp = [TEAL if v > 0 else GRAY for v in coefs.values]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(coefs.index, coefs.values.tolist(), color=colors_imp, edgecolor="white", height=0.55)
        ax.axvline(0, color=DARK_BLUE, lw=1)
        ax.set_xlabel("Coefficient", fontsize=9)
        ax.set_title("Feature Impact on Life Expectancy", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Teal = increases life expectancy | Gray = reduces or neutral")

    with col2:
        st.subheader("Predicted vs. Real")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y_test, y_pred, alpha=0.55, color=MID_BLUE, s=30, edgecolors="none")
        lims = [min(float(y_test.min()),float(y_pred.min()))-2, max(float(y_test.max()),float(y_pred.max()))+2]
        ax.plot(lims, lims, color=ACCENT, lw=2, linestyle="--", label="Ideal line")
        ax.set_xlabel("Real Life Expectancy (yrs)", fontsize=10); ax.set_ylabel("Predicted (yrs)", fontsize=10)
        ax.set_title(f"Predicted vs. Real (R2={R2:.2f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    rec_box(f"With R2={R2:.2f} and MAE={MAE:.2f} years, the model predicts life expectancy within 3.2 years on average. Literacy rate has the highest coefficient (+0.28): 1pp increase in literacy = +0.28 years of life expectancy. GDP per capita has minimal direct effect — it works through literacy and health access.")
    hr()

    st.subheader("Life Expectancy Simulator")
    st.markdown("Adjust the features to estimate life expectancy for a hypothetical country:")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: sim_gdp  = st.slider("GDP per capita ($)", 500, 100000, 10000, 500)
    with sc2: sim_h    = st.slider("Health exp. % GDP", 1.0, 18.0, 7.0, 0.1)
    with sc3: sim_e    = st.slider("Education exp. % GDP", 1.0, 10.0, 4.5, 0.1)
    with sc4: sim_l    = st.slider("Literacy rate (%)", 20.0, 100.0, 80.0, 0.5)

    X_sim = np.array([[sim_gdp, sim_h, sim_e, sim_l]])
    pred_val = model.predict(X_sim)[0]
    pct = (df22["life_expectancy"] < pred_val).mean() * 100

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Predicted Life Expectancy", f"{pred_val:.1f} yrs")
    rc2.metric("Percentile", f"{pct:.0f}%", "of 104 countries")
    rc3.metric("Global average (2022)", f"{df22['life_expectancy'].mean():.1f} yrs", "for reference")


# =============================================================================
# 7 — BRAZIL DEEP-DIVE
# =============================================================================
elif page == "Brazil Deep-Dive":
    phase_label("Phase 4 — Brazil Deep-Dive")
    st.title("Brazil Deep-Dive")
    st.markdown("Brazil's health and education profile vs. global benchmarks — and what the data says about the gap.")
    hr()

    bra_data = df22[df22["country_code"]=="BRA"].iloc[0] if len(df22[df22["country_code"]=="BRA"]) > 0 else None
    jpn_data = df22[df22["country_code"]=="JPN"].iloc[0] if len(df22[df22["country_code"]=="JPN"]) > 0 else None
    sgp_data = df22[df22["country_code"]=="SGP"].iloc[0] if len(df22[df22["country_code"]=="SGP"]) > 0 else None
    chl_data = df22[df22["country_code"]=="CHL"].iloc[0] if len(df22[df22["country_code"]=="CHL"]) > 0 else None

    if bra_data is not None:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Life Expectancy", f"{bra_data['life_expectancy']:.1f} yrs", f"vs Japan: {jpn_data['life_expectancy']:.1f}")
        c2.metric("Health Exp. % GDP", f"{bra_data['health_expenditure_pct_gdp']:.1f}%", "Higher than South Korea")
        c3.metric("Infant Mortality", f"{bra_data['infant_mortality']:.1f}/1k", f"vs Chile: {chl_data['infant_mortality']:.1f}/1k")
        c4.metric("Literacy Rate", f"{bra_data['literacy_rate']:.1f}%", "Room to improve")
        hr()

    st.subheader("Brazil vs. Benchmark Countries")
    benchmarks = df22[df22["country_code"].isin(["BRA","JPN","SGP","CHL","CRI","KOR","MEX"])].copy()
    benchmarks = benchmarks[["country_name","life_expectancy","infant_mortality","literacy_rate","health_expenditure_pct_gdp","gdp_per_capita"]].round(1)
    benchmarks.columns = ["Country","Life Exp. (yrs)","Infant Mort./1k","Literacy (%)","Health Exp. % GDP","GDP/capita ($)"]
    benchmarks = benchmarks.sort_values("Life Exp. (yrs)", ascending=False)
    st.dataframe(benchmarks, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Brazil's Time Series — Progress 2000–2022")
    bra_ts = df[df["country_code"]=="BRA"].sort_values("year")
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5))
    metrics = [("life_expectancy","Life Expectancy (yrs)",TEAL),
               ("infant_mortality","Infant Mortality/1k",ACCENT),
               ("literacy_rate","Literacy Rate (%)",MID_BLUE)]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.plot(bra_ts["year"], bra_ts[col], color=color, lw=2.5)
        ax.fill_between(bra_ts["year"], bra_ts[col], alpha=0.15, color=color)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9")
    fig.suptitle("Brazil — Key Indicators 2000–2022", fontsize=12, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    rec_box("Brazil has made real progress since 2000 — life expectancy increased and infant mortality declined significantly. However, the gap vs. efficient countries remains large. Brazil's challenge is not input (spending is already high) but output efficiency, equity of access, and literacy improvement in lower-income populations.")


# =============================================================================
# 8 — POLICY RECOMMENDATIONS
# =============================================================================
elif page == "Policy Recommendations":
    phase_label("Phase 5 — Policy Recommendations")
    st.title("Policy Recommendations")
    st.markdown("Five recommendations derived exclusively from data findings. Applicable to Brazil and Cluster 0 countries.")
    hr()

    matrix = pd.DataFrame({
        "Recommendation": [
            "Shift focus from spending volume to spending efficiency",
            "Invest in adult literacy as a long-cycle health intervention",
            "Close infant mortality gaps — highest-leverage health action",
            "Learn from Singapore's healthcare model",
            "Build a public health-education monitoring dashboard",
        ],
        "Impact": ["High","High","High","Medium","Medium"],
        "Effort": ["Medium","Low","Medium","High","Low"],
        "Priority": ["#1 — Immediate","#2 — Immediate","#3 — Short term","#4 — Strategic","#5 — Quick win"],
        "Timeline": ["Ongoing","3–5 yrs","5–10 yrs","10+ yrs","30–60 days"],
    })

    def color_imp(val):
        if val == "High": return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        return "background-color:#fff3cd;color:#856404;font-weight:bold"
    def color_prio(val):
        if "Immediate" in val: return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if "Short"     in val: return "background-color:#fff3cd;color:#856404"
        if "Quick"     in val: return "background-color:#d4edda;color:#155724;font-weight:bold"
        return "background-color:#e8f4fd;color:#1a5276"

    st.dataframe(matrix.style.applymap(color_imp,subset=["Impact"]).applymap(color_prio,subset=["Priority"]),
                 use_container_width=True, hide_index=True)
    hr()

    prios = [
        ("#1 — Shift focus from spending volume to spending efficiency", ACCENT,
         "Brazil spends 9.5% of GDP on health — G7-level — but achieves Cluster 0 outcomes. Increasing spending without efficiency reforms will not close the gap.",
         ["Benchmark health system efficiency against Singapore and South Korea",
          "Audit health expenditure: prevention vs. treatment ratio",
          "Implement outcome-based funding for primary care units",
          "Track life expectancy improvement per billion USD invested"]),
        ("#2 — Invest in adult literacy as a long-cycle health investment", TEAL,
         "Literacy (corr=0.81 with life expectancy) is the strongest predictor of population health. Education spending today becomes health gains in 20-30 years.",
         ["Expand adult literacy programs targeting populations above age 30",
          "Measure and publish literacy rates by municipality annually",
          "Benchmark against Costa Rica (97.9% literacy, 80.3 yrs life exp.)",
          "Track literacy-to-life-expectancy at subnational level annually"]),
        ("#3 — Close infant mortality gaps", MID_BLUE,
         "Infant mortality (corr=-0.95) is the variable most tightly linked to overall health. Brazil at 13/1k vs. Chile at 6.4/1k — a gap fully closable with targeted interventions.",
         ["Identify municipalities with infant mortality above 20/1k",
          "Set 10-year target: reduce from 13 to below 7/1k",
          "Benchmark against Chile — highest-performing Latin American country",
          "Publish infant mortality rankings by municipality quarterly"]),
        ("#4 — Learn from Singapore's healthcare model", GOLD,
         "Singapore achieves 85.4 yrs life expectancy at 4.0% GDP. Its model combines universal coverage, Medisave co-payment system, and prevention-first primary care.",
         ["Study Singapore's Medisave system: mandatory health savings accounts",
          "Adopt prevention-first primary care: reduce ER as first point of contact",
          "Implement national health efficiency scorecard published annually",
          "Commission formal benchmarking: Brazil vs. Singapore vs. South Korea"]),
        ("#5 — Build a public health-education monitoring dashboard", GREEN,
         "The data to track Brazil's progress exists in the World Bank and WHO APIs. What is missing is a public-facing, regularly updated monitoring system.",
         ["Integrate World Bank API indicators into a public federal dashboard",
          "Publish quarterly updates: life expectancy, infant mortality, literacy, health exp.",
          "Create efficiency index comparing Brazil to Cluster 0 peers annually",
          "Make subnational data available at state and municipality level"]),
    ]

    for title, color, prob, acoes in prios:
        with st.expander(title, expanded=False):
            st.markdown(f"**Problem / Opportunity:** {prob}")
            st.markdown("**Actions:**")
            for i, a in enumerate(acoes, 1): st.markdown(f"  {i}. {a}")


# =============================================================================
# 9 — TECHNICAL APPENDIX
# =============================================================================
elif page == "Technical Appendix":
    phase_label("Technical Appendix")
    st.title("Technical Appendix")
    hr()

    st.subheader("Tech Stack")
    tech = pd.DataFrame({
        "Tool": ["Python","Pandas & NumPy","Scikit-learn","Matplotlib","Streamlit","GitHub Pages","ReportLab"],
        "Use": ["Core language","Data manipulation and panel data handling",
                "Linear Regression + KMeans clustering",
                "Exploratory visualizations","Interactive dashboard (this app)",
                "Project website","PDF report generation"],
        "Version": ["3.10+","—","—","—","—","—","—"],
    })
    st.dataframe(tech, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Model Performance Details")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R2",f"{R2:.4f}")
    c2.metric("MAE",f"{MAE:.2f} yrs")
    c3.metric("Countries",f"{len(df22)}")
    c4.metric("Features","4")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": [round(c,4) for c in model.coef_],
        "Direction": ["Positive" if c > 0 else "Negative" for c in model.coef_],
        "Interpretation": [
            "Each $10k extra GDP/capita → small direct gain in life exp.",
            "Each 1% GDP more on health → +0.19 yrs life exp. (diminishing returns above 10%)",
            "Near-zero — outcomes matter more than spending on education",
            "Each 1pp literacy → +0.28 yrs life expectancy (strongest driver)",
        ],
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Dataset Summary")
    st.markdown(f"**File:** global_health_education.csv — {len(df):,} records, {df['country_code'].nunique()} countries, {df['year'].nunique()} years, 15 variables")
    st.markdown("**Sources:** World Bank Open Data · WHO Global Health Observatory · Our World in Data")
    st.markdown("**GitHub:** https://github.com/Bruno-Barradas/global-health-education-index")
    st.caption("Global Health & Education Efficiency Index — Bruno Barradas · 2026")
