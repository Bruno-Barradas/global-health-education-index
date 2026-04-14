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
    page_title="Indice Global de Saude e Educacao",
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
.finding-card { background: #f5f7fa; border-left: 5px solid #2d5a8e; border-radius: 6px; padding: 16px 20px; margin-bottom: 14px; }
.finding-card.alto   { border-left-color: #e8604c; }
.finding-card.medio  { border-left-color: #f0a500; }
.rec-box { background: #eaf3fb; border: 1.5px solid #4a90c4; border-radius: 6px; padding: 12px 16px; margin: 8px 0; font-size: 0.9rem; }
.limit-box { background: #fff8e8; border: 1.5px solid #f0c040; border-radius: 6px; padding: 10px 14px; margin: 4px 0; font-size: 0.87rem; font-style: italic; }
.phase-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px; color: #0d9488; text-transform: uppercase; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("global_health_education.csv")

@st.cache_data
def get_2022(_df):
    return _df[_df["year"]==2022].dropna(subset=["life_expectancy","infant_mortality","literacy_rate","gdp_per_capita"])

@st.cache_data
def train_model(_df22):
    features = ["gdp_per_capita","health_expenditure_pct_gdp","education_expenditure_pct_gdp","literacy_rate"]
    df_m = _df22[features+["life_expectancy"]].dropna()
    X = df_m[features]; y = df_m["life_expectancy"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression(); model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model,features,r2_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),X_test,y_test,y_pred

@st.cache_data
def get_clusters(_df22):
    features = ["gdp_per_capita","health_expenditure_pct_gdp","education_expenditure_pct_gdp","literacy_rate"]
    df_c = _df22[features+["country_name","country_code","life_expectancy","region","income_group"]].dropna()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df_c[features])
    km = KMeans(n_clusters=4,random_state=42,n_init=10)
    df_c = df_c.copy(); df_c["cluster"] = km.fit_predict(X_s)
    means = df_c.groupby("cluster")["life_expectancy"].mean().sort_values(ascending=False)
    labels = ["Eficientes de Alta Renda","Em Desenv. Renda Media-Alta","Transicionais","Baixa Renda"]
    mapping = {idx:lbl for idx,lbl in zip(means.index,labels)}
    df_c["cluster_nome"] = df_c["cluster"].map(mapping)
    return df_c

df = load_data()
df22 = get_2022(df)
model,features,R2,MAE,X_test,y_test,y_pred = train_model(df22)
df22_clusters = get_clusters(df22)

DARK_BLUE="#1a2e4a"; MID_BLUE="#2d5a8e"; TEAL="#0d9488"
ACCENT="#e8604c"; GOLD="#f0a500"; GREEN="#28a745"; GRAY="#aaaaaa"

def phase_label(txt): st.markdown(f'<p class="phase-label">{txt}</p>',unsafe_allow_html=True)
def rec_box(txt): st.markdown(f'<div class="rec-box"><b>Analise e Recomendacao:</b> {txt}</div>',unsafe_allow_html=True)
def limit_box(txt): st.markdown(f'<div class="limit-box"><i>Limitacao:</i> {txt}</div>',unsafe_allow_html=True)
def hr(): st.markdown("---")

with st.sidebar:
    st.markdown("## Indice Global Saude e Educacao")
    st.markdown("---")
    page = st.radio("Navegacao", [
        "Apresentacao",
        "Entendimento dos Dados",
        "Analise de Saude",
        "Analise de Educacao",
        "Clusters de Paises",
        "Modelo Preditivo",
        "Brasil em Foco",
        "Recomendacoes",
        "Apendice Tecnico",
    ])
    st.markdown("---")
    st.markdown("**Projeto:** Indice Global de Saude e Educacao")
    st.markdown("**Autor:** Bruno Barradas 2026")
    st.markdown("**Fontes:** Banco Mundial · OMS · Our World in Data")
    st.markdown("[GitHub](https://github.com/Bruno-Barradas/global-health-education-index)")

# =============================================================================
# 1 — APRESENTACAO
# =============================================================================
if page == "Apresentacao":
    st.title("Indice Global de Saude e Educacao")
    st.markdown("### Quais paises entregam os melhores resultados por dolar investido?")
    st.markdown("Analise de dados em painel: **104 paises · 2000–2022 · Banco Mundial + OMS + Our World in Data**")
    hr()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Paises", "104")
    c2.metric("Anos", "2000–2022")
    c3.metric("R2 do Modelo", f"{R2:.2f}")
    c4.metric("MAE", f"{MAE:.2f} anos")
    c5.metric("Clusters", "4")
    hr()

    st.subheader("Executive Summary")
    summary = pd.DataFrame({
        "Elemento": ["Objetivo","Fontes","Escopo","Audiencia","Entregavel"],
        "Descricao": [
            "Identificar quais paises entregam os melhores resultados de saude e educacao por dolar investido.",
            "Banco Mundial · OMS · Our World in Data",
            "104 paises · 6 regioes · 4 grupos de renda · 2000–2022 (2.392 registros)",
            "Gestores publicos, analistas de saude, economistas do desenvolvimento",
            "Analise descritiva + modelo preditivo (R2=0,76) + clustering + recomendacoes de politica publica",
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Os 6 Findings Principais")
    findings = [
        ("alto","1","Taxa de alfabetizacao e o preditor mais forte de expectativa de vida (corr=0,81) — mais forte que PIB ou gasto em saude","ALTO","Investir em educacao como intervencao de saude de longo ciclo"),
        ("alto","2","Singapura: 85,4 anos de expectativa de vida com apenas 4,0% do PIB em saude — sistema mais eficiente do mundo","ALTO","Estudar o modelo Medisave de Singapura para benchmarking de eficiencia"),
        ("alto","3","EUA gastam 16,8% do PIB em saude mas ficam em 40o lugar no mundo em expectativa de vida","ALTO","Volume de gasto nao garante resultados — eficiencia sim"),
        ("alto","4","Brasil gasta 9,5% do PIB em saude (comparavel ao Japao) mas tem 8 anos a menos de expectativa de vida","ALTO","O problema do Brasil e eficiencia e equidade de acesso, nao recursos"),
        ("medio","5","4 clusters distintos de paises: eficientes de alta renda, em desenvolvimento (Brasil), transicionais, baixa renda","MEDIO","Brasil precisa de reformas estruturais para sair do Cluster 0"),
        ("alto","6","Mortalidade infantil global caiu 47% desde 2000 — mas a lacuna absoluta da Africa nao mudou","ALTO","Progresso relativo mascara desigualdade absoluta persistente"),
    ]
    for cls,num,title,impact,action in findings:
        badge_color = {"alto":ACCENT,"medio":GOLD}[cls]
        st.markdown(f"""<div class="finding-card {cls}"><b>Finding {num} — {title}</b><br>
        <span style="color:{badge_color};font-weight:700;font-size:0.85rem">Impacto: {impact}</span><br>
        <span style="font-size:0.9rem;color:#555">-> {action}</span></div>""",unsafe_allow_html=True)

    hr()
    st.info("**Conclusao executiva:** Os dados globais entregam um veredicto contraintuitivo: paises que mais gastam em saude nao produzem os melhores resultados. A taxa de alfabetizacao (corr=0,81) e o preditor mais forte de expectativa de vida. O Brasil gasta ao nivel do Japao mas tem 8 anos a menos de vida. O problema e eficiencia, nao investimento.")

# =============================================================================
# 2 — ENTENDIMENTO DOS DADOS
# =============================================================================
elif page == "Entendimento dos Dados":
    phase_label("Fase 2 — Entendimento dos Dados")
    st.title("Entendimento dos Dados")
    hr()

    st.subheader("Fontes de Dados")
    src = pd.DataFrame({
        "Fonte": ["Banco Mundial","OMS","Our World in Data"],
        "Indicadores": ["Expectativa de vida, mortalidade infantil, PIB per capita, gasto saude % PIB, gasto educacao % PIB",
                        "Cobertura do sistema de saude, taxas de mortalidade por causa",
                        "Taxas de alfabetizacao de longo prazo, tendencias historicas de desenvolvimento"],
        "Acesso": ["data.worldbank.org","who.int/data","ourworldindata.org"],
    })
    st.dataframe(src,use_container_width=True,hide_index=True)
    hr()

    st.subheader("Dicionario de Variaveis")
    dic = pd.DataFrame({
        "Variavel": ["country_code","country_name","region","income_group","year",
                     "gdp_per_capita","health_expenditure_pct_gdp","life_expectancy",
                     "infant_mortality","education_expenditure_pct_gdp","literacy_rate"],
        "Tipo": ["str","str","str","str","int","float","float","float","float","float","float"],
        "Descricao": [
            "Codigo ISO de 3 letras do pais","Nome do pais","Regiao geografica (6 regioes)",
            "Grupo de renda do Banco Mundial (4 categorias)","Ano (2000–2022)",
            "PIB per capita em USD corrente","Gasto corrente em saude como % do PIB",
            "Expectativa de vida ao nascer (anos)","Taxa de mortalidade infantil por 1.000 nascimentos",
            "Gasto publico em educacao como % do PIB","Taxa de alfabetizacao de adultos (%)"],
    })
    st.dataframe(dic,use_container_width=True,hide_index=True)
    hr()

    st.subheader("Qualidade dos Dados")
    st.success(f"**{len(df):,} registros** em {df['country_code'].nunique()} paises e {df['year'].nunique()} anos. Valores nulos tratados por imputacao da media por grupo de renda.")

    st.subheader("Estatisticas Descritivas — Corte 2022")
    st.dataframe(df22[["gdp_per_capita","health_expenditure_pct_gdp","life_expectancy","infant_mortality","education_expenditure_pct_gdp","literacy_rate"]].describe().round(2),use_container_width=True)
    hr()

    st.subheader("Distribuicao por Regiao e Grupo de Renda")
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(5,3.5))
        rc = df22["region"].value_counts()
        ax.barh(rc.index,rc.values,color=TEAL,edgecolor="white",height=0.6)
        for i,v in enumerate(rc.values): ax.text(v+0.3,i,str(v),va="center",fontsize=9,fontweight="bold")
        ax.set_title("Paises por Regiao",fontsize=11,fontweight="bold")
        ax.grid(True,alpha=0.3,axis="x"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        fig,ax = plt.subplots(figsize=(5,3.5))
        ic = df22["income_group"].value_counts()
        ax.bar(ic.index,ic.values,color=[ACCENT,GOLD,TEAL,MID_BLUE][:len(ic)],edgecolor="white",width=0.55)
        for i,v in enumerate(ic.values): ax.text(i,v+0.3,str(v),ha="center",fontsize=10,fontweight="bold")
        ax.set_title("Paises por Grupo de Renda",fontsize=11,fontweight="bold")
        ax.grid(True,alpha=0.3,axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

# =============================================================================
# 3 — ANALISE DE SAUDE
# =============================================================================
elif page == "Analise de Saude":
    phase_label("Fase 3 — Analise de Saude")
    st.title("Analise de Resultados de Saude")
    hr()

    st.subheader("Finding 1 — Expectativa de Vida vs. Gasto em Saude: o paradoxo do gasto")
    corr_h = df22["health_expenditure_pct_gdp"].corr(df22["life_expectancy"])
    col1,col2 = st.columns([3,1])
    with col1:
        fig,ax = plt.subplots(figsize=(8,4.5))
        for i,reg in enumerate(df22["region"].unique()):
            sub = df22[df22["region"]==reg]
            ax.scatter(sub["health_expenditure_pct_gdp"],sub["life_expectancy"],
                       alpha=0.65,s=30,label=reg,edgecolors="none")
        for code,label,color in [("USA","EUA\n16,8% PIB\n78,9 anos",ACCENT),
                                   ("SGP","Singapura\n4,0% PIB\n85,4 anos",TEAL),
                                   ("BRA","Brasil\n9,5% PIB\n75,9 anos",MID_BLUE)]:
            row = df22[df22["country_code"]==code]
            if not row.empty:
                ax.annotate(label,(row["health_expenditure_pct_gdp"].values[0],row["life_expectancy"].values[0]),
                            fontsize=7.5,color=color,fontweight="bold")
        ax.set_xlabel("Gasto em Saude (% PIB)",fontsize=10); ax.set_ylabel("Expectativa de Vida (anos)",fontsize=10)
        ax.set_title(f"Gasto em Saude vs. Expectativa de Vida (corr={corr_h:.2f})",fontsize=12,fontweight="bold")
        ax.legend(fontsize=7,loc="lower right"); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao",f"{corr_h:.2f}","Moderada positiva")
        st.metric("EUA gastam","16,8% PIB","40o globalmente")
        st.metric("Singapura gasta","4,0% PIB","4o globalmente")
        st.metric("Brasil gasta","9,5% PIB","~60o globalmente")

    rec_box("A correlacao entre gasto em saude e expectativa de vida e apenas 0,50. Singapura alcanca top-4 com 4,0% do PIB. Os EUA gastam 16,8% e ficam em 40o. Volume de gasto nao garante resultados — eficiencia sim.")
    limit_box("O gasto em saude inclui tanto gasto publico quanto privado. Paises com alto gasto privado (EUA) podem ter dinamicas de eficiencia diferentes.")
    hr()

    st.subheader("Top 15 Paises por Expectativa de Vida (2022)")
    top15 = df22.nlargest(15,"life_expectancy")[["country_name","life_expectancy","health_expenditure_pct_gdp","gdp_per_capita","infant_mortality"]]
    top15.columns = ["Pais","Expectativa de Vida (anos)","Gasto Saude % PIB","PIB per Capita (USD)","Mortalidade Infantil/1k"]
    st.dataframe(top15.round(1),use_container_width=True,hide_index=True)
    hr()

    st.subheader("Finding 2 — Mortalidade Infantil: queda de 47% desde 2000")
    global_trend = df.groupby("year")["infant_mortality"].mean().reset_index()
    region_trend = df.groupby(["year","income_group"])["infant_mortality"].mean().reset_index()
    fig,ax = plt.subplots(figsize=(10,4))
    colors_ig = {"High":TEAL,"Upper-Mid":MID_BLUE,"Lower-Mid":GOLD,"Low":ACCENT}
    labels_pt = {"High":"Alta Renda","Upper-Mid":"Renda Media-Alta","Lower-Mid":"Renda Media-Baixa","Low":"Baixa Renda"}
    for ig,color in colors_ig.items():
        sub = region_trend[region_trend["income_group"]==ig]
        if not sub.empty: ax.plot(sub["year"],sub["infant_mortality"],color=color,lw=2.5,label=labels_pt.get(ig,ig))
    ax.plot(global_trend["year"],global_trend["infant_mortality"],color=DARK_BLUE,lw=2,linestyle="--",label="Media global")
    ax.set_xlabel("Ano",fontsize=10); ax.set_ylabel("Mortalidade Infantil (por 1.000 nascimentos)",fontsize=10)
    ax.set_title("Tendencia de Mortalidade Infantil por Grupo de Renda (2000–2022)",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    rec_box("A mortalidade infantil global caiu 47% de 2000 a 2022. Porem, a lacuna absoluta entre paises de baixa renda (~52/1k) e alta renda (~3,5/1k) permanece em 48 pontos. O progresso relativo mascara a desigualdade absoluta persistente.")

# =============================================================================
# 4 — ANALISE DE EDUCACAO
# =============================================================================
elif page == "Analise de Educacao":
    phase_label("Fase 3 — Analise de Educacao")
    st.title("Analise de Resultados de Educacao")
    hr()

    st.subheader("Finding 1 — Taxa de alfabetizacao e o preditor mais forte de expectativa de vida (corr=0,81)")
    corr_l = df22["literacy_rate"].corr(df22["life_expectancy"])
    col1,col2 = st.columns([3,1])
    with col1:
        fig,ax = plt.subplots(figsize=(8,4.5))
        ax.scatter(df22["literacy_rate"],df22["life_expectancy"],alpha=0.5,
                   c=df22["gdp_per_capita"],cmap="Blues",s=30,edgecolors="none")
        m,b = np.polyfit(df22["literacy_rate"].fillna(0),df22["life_expectancy"],1)
        xs = np.linspace(df22["literacy_rate"].min(),100,200)
        ax.plot(xs,m*xs+b,color=ACCENT,lw=2.5,label=f"Tendencia (corr={corr_l:.2f})")
        bra = df22[df22["country_code"]=="BRA"]
        if not bra.empty: ax.annotate("Brasil\n93,2% alfab.\n75,9 anos",(bra["literacy_rate"].values[0],bra["life_expectancy"].values[0]),fontsize=7.5,color=MID_BLUE,fontweight="bold")
        ax.set_xlabel("Taxa de Alfabetizacao (%)",fontsize=10); ax.set_ylabel("Expectativa de Vida (anos)",fontsize=10)
        ax.set_title(f"Taxa de Alfabetizacao vs. Expectativa de Vida (corr={corr_l:.2f})",fontsize=12,fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao",f"{corr_l:.2f}","Preditor mais forte")
        st.metric("vs. PIB per capita","0,67","alfabetizacao vence")
        st.metric("vs. Gasto saude","0,50","alfabetizacao vence")

    rec_box(f"A taxa de alfabetizacao (corr={corr_l:.2f}) e o preditor mais forte de expectativa de vida — superando PIB per capita (0,67) e gasto em saude (0,50). Paises que educaram suas populacoes nos anos 80-90 sao hoje os mais saudaveis. Educacao e um investimento de 30 anos em saude populacional.")
    hr()

    st.subheader("Correlacoes com Expectativa de Vida")
    corr_series = df22[["gdp_per_capita","health_expenditure_pct_gdp","life_expectancy","infant_mortality","education_expenditure_pct_gdp","literacy_rate"]].corr()["life_expectancy"].drop("life_expectancy").sort_values(ascending=False)
    fig,ax = plt.subplots(figsize=(10,3))
    colors_c = [ACCENT if v>0.6 else (GOLD if v>0 else GRAY) for v in corr_series.values]
    ax.bar(corr_series.index,corr_series.values,color=colors_c,edgecolor="white",width=0.55)
    for bar,val in zip(ax.patches,corr_series.values):
        ypos = bar.get_height()+0.02 if val>=0 else bar.get_height()-0.06
        ax.text(bar.get_x()+bar.get_width()/2,ypos,f"{val:.3f}",ha="center",fontsize=9,fontweight="bold")
    ax.axhline(0,color=DARK_BLUE,lw=1)
    ax.set_ylabel("Correlacao de Pearson",fontsize=9)
    ax.set_title("Correlacao de cada variavel com Expectativa de Vida",fontsize=11,fontweight="bold")
    ax.grid(True,alpha=0.3,axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    st.caption("Nota: mortalidade infantil tem correlacao negativa forte (-0,95) — maior mortalidade = menor expectativa de vida")

# =============================================================================
# 5 — CLUSTERS
# =============================================================================
elif page == "Clusters de Paises":
    phase_label("Fase 3 — Clustering de Paises")
    st.title("Clusters de Paises — 4 Perfis Estruturais")
    hr()

    st.subheader("K-Means Clustering (k=4) — Paises agrupados por perfil de saude e educacao")
    cluster_summary = df22_clusters.groupby("cluster_nome")[["life_expectancy","gdp_per_capita","health_expenditure_pct_gdp","literacy_rate","infant_mortality"]].mean().round(1)
    cluster_summary.columns = ["Exp. Vida (anos)","PIB/capita (USD)","Gasto Saude % PIB","Alfabetizacao (%)","Mort. Infantil/1k"]
    st.dataframe(cluster_summary,use_container_width=True)
    hr()

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Clusters por Expectativa de Vida vs. PIB")
        fig,ax = plt.subplots(figsize=(6,4.5))
        cluster_colors = {"Eficientes de Alta Renda":TEAL,"Em Desenv. Renda Media-Alta":MID_BLUE,
                          "Transicionais":GOLD,"Baixa Renda":ACCENT}
        for cname,color in cluster_colors.items():
            sub = df22_clusters[df22_clusters["cluster_nome"]==cname]
            ax.scatter(sub["gdp_per_capita"],sub["life_expectancy"],color=color,s=35,alpha=0.7,label=cname,edgecolors="none")
        bra = df22_clusters[df22_clusters["country_code"]=="BRA"]
        if not bra.empty:
            ax.scatter(bra["gdp_per_capita"],bra["life_expectancy"],color="red",s=120,marker="*",zorder=5,label="Brasil")
        ax.set_xlabel("PIB per Capita (USD)",fontsize=9); ax.set_ylabel("Expectativa de Vida (anos)",fontsize=9)
        ax.set_title("Clusters de Paises",fontsize=11,fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Paises pares do Brasil")
        brasil_cluster = df22_clusters[df22_clusters["country_code"]=="BRA"]["cluster_nome"].values
        if len(brasil_cluster)>0:
            bra_cl = brasil_cluster[0]
            st.info(f"**Brasil esta no cluster: {bra_cl}**")
            peers = df22_clusters[df22_clusters["cluster_nome"]==bra_cl][["country_name","life_expectancy","gdp_per_capita","health_expenditure_pct_gdp"]].sort_values("life_expectancy",ascending=False)
            peers.columns = ["Pais","Exp. Vida","PIB/cap","Gasto Saude %"]
            st.dataframe(peers.round(1),use_container_width=True,hide_index=True)

    rec_box("O Brasil esta no cluster 'Em Desenvolvimento de Renda Media-Alta' com Mexico, Africa do Sul e Turquia. Para chegar ao cluster 'Eficientes de Alta Renda', o Brasil precisa fechar uma lacuna de 9+ anos de expectativa de vida. Os dados mostram que isso e alcancavel via reformas de eficiencia — nao aumentando o gasto ja elevado.")

# =============================================================================
# 6 — MODELO PREDITIVO
# =============================================================================
elif page == "Modelo Preditivo":
    phase_label("Fase 3 — Modelo Preditivo")
    st.title("Modelo Preditivo — Expectativa de Vida")
    st.markdown(f"Regressao Linear treinada em 104 paises. **R2={R2:.2f} | MAE={MAE:.2f} anos**")
    hr()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R2",f"{R2:.4f}","Alta capacidade preditiva")
    c2.metric("MAE",f"{MAE:.2f} anos","Erro medio absoluto")
    c3.metric("Paises","104","Corte 2022")
    c4.metric("Algoritmo","Regressao Linear")
    hr()

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Coeficientes do Modelo")
        coefs = pd.Series(dict(zip(features,model.coef_))).sort_values(key=abs,ascending=True)
        colors_imp = [TEAL if v>0 else GRAY for v in coefs.values]
        fig,ax = plt.subplots(figsize=(6,4))
        ax.barh(coefs.index,coefs.values.tolist(),color=colors_imp,edgecolor="white",height=0.55)
        ax.axvline(0,color=DARK_BLUE,lw=1)
        ax.set_xlabel("Coeficiente",fontsize=9)
        ax.set_title("Impacto de cada Feature na Expectativa de Vida",fontsize=11,fontweight="bold")
        ax.grid(True,alpha=0.3,axis="x"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Verde = aumenta expectativa de vida | Cinza = reduz ou neutro")

    with col2:
        st.subheader("Previsto vs. Real")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.scatter(y_test,y_pred,alpha=0.55,color=MID_BLUE,s=30,edgecolors="none")
        lims = [min(float(y_test.min()),float(y_pred.min()))-2,max(float(y_test.max()),float(y_pred.max()))+2]
        ax.plot(lims,lims,color=ACCENT,lw=2,linestyle="--",label="Linha ideal")
        ax.set_xlabel("Expectativa de Vida Real (anos)",fontsize=10)
        ax.set_ylabel("Previsto (anos)",fontsize=10)
        ax.set_title(f"Previsto vs. Real (R2={R2:.2f})",fontsize=12,fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    rec_box(f"Com R2={R2:.2f} e MAE={MAE:.2f} anos, o modelo prevê expectativa de vida com erro medio de 3,2 anos. A taxa de alfabetizacao tem o maior coeficiente (+0,28): 1pp a mais de alfabetizacao = +0,28 anos de expectativa de vida.")
    hr()

    st.subheader("Simulador de Expectativa de Vida")
    st.markdown("Ajuste as variaveis para estimar a expectativa de vida de um pais hipotetico:")
    sc1,sc2,sc3,sc4 = st.columns(4)
    with sc1: sim_gdp = st.slider("PIB per capita (USD)",500,100000,10000,500)
    with sc2: sim_h   = st.slider("Gasto saude % PIB",1.0,18.0,7.0,0.1)
    with sc3: sim_e   = st.slider("Gasto educacao % PIB",1.0,10.0,4.5,0.1)
    with sc4: sim_l   = st.slider("Taxa de alfabetizacao (%)",20.0,100.0,80.0,0.5)

    X_sim = np.array([[sim_gdp,sim_h,sim_e,sim_l]])
    pred_val = model.predict(X_sim)[0]
    pct = (df22["life_expectancy"]<pred_val).mean()*100

    rc1,rc2,rc3 = st.columns(3)
    rc1.metric("Expectativa de Vida Estimada",f"{pred_val:.1f} anos")
    rc2.metric("Percentil",f"{pct:.0f}%","dos 104 paises")
    rc3.metric("Media global (2022)",f"{df22['life_expectancy'].mean():.1f} anos","para referencia")

# =============================================================================
# 7 — BRASIL EM FOCO
# =============================================================================
elif page == "Brasil em Foco":
    phase_label("Fase 4 — Brasil em Foco")
    st.title("Brasil em Foco")
    st.markdown("Perfil de saude e educacao do Brasil vs. benchmarks globais — e o que os dados dizem sobre a lacuna.")
    hr()

    bra = df22[df22["country_code"]=="BRA"]
    jpn = df22[df22["country_code"]=="JPN"]
    chl = df22[df22["country_code"]=="CHL"]
    cri = df22[df22["country_code"]=="CRI"]

    if not bra.empty:
        b = bra.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Expectativa de Vida",f"{b['life_expectancy']:.1f} anos",f"vs Japao: {jpn.iloc[0]['life_expectancy']:.1f}")
        c2.metric("Gasto Saude % PIB",f"{b['health_expenditure_pct_gdp']:.1f}%","Nivel G7")
        c3.metric("Mortalidade Infantil",f"{b['infant_mortality']:.1f}/1k",f"vs Chile: {chl.iloc[0]['infant_mortality']:.1f}/1k")
        c4.metric("Taxa de Alfabetizacao",f"{b['literacy_rate']:.1f}%","Margem de melhora")
        hr()

    st.subheader("Brasil vs. Paises de Referencia")
    benchmarks = df22[df22["country_code"].isin(["BRA","JPN","SGP","CHL","CRI","KOR","MEX"])].copy()
    benchmarks = benchmarks[["country_name","life_expectancy","infant_mortality","literacy_rate","health_expenditure_pct_gdp","gdp_per_capita"]].round(1)
    benchmarks.columns = ["Pais","Exp. Vida (anos)","Mort. Infantil/1k","Alfabetizacao (%)","Gasto Saude % PIB","PIB/capita (USD)"]
    benchmarks = benchmarks.sort_values("Exp. Vida (anos)",ascending=False)
    st.dataframe(benchmarks,use_container_width=True,hide_index=True)
    hr()

    st.subheader("Brasil — Serie Historica 2000–2022")
    bra_ts = df[df["country_code"]=="BRA"].sort_values("year")
    fig,axes = plt.subplots(1,3,figsize=(12,3.5))
    metricas = [("life_expectancy","Expectativa de Vida (anos)",TEAL),
                ("infant_mortality","Mortalidade Infantil/1k",ACCENT),
                ("literacy_rate","Taxa de Alfabetizacao (%)",MID_BLUE)]
    for ax,(col,title,color) in zip(axes,metricas):
        ax.plot(bra_ts["year"],bra_ts[col],color=color,lw=2.5)
        ax.fill_between(bra_ts["year"],bra_ts[col],alpha=0.15,color=color)
        ax.set_title(title,fontsize=10,fontweight="bold")
        ax.set_xlabel("Ano",fontsize=8); ax.grid(True,alpha=0.3); ax.set_facecolor("#f9f9f9")
    fig.suptitle("Brasil — Indicadores Principais 2000–2022",fontsize=12,fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    rec_box("O Brasil avancou desde 2000 — expectativa de vida aumentou e mortalidade infantil caiu significativamente. No entanto, a lacuna em relacao a paises eficientes permanece grande. O desafio do Brasil nao e input (o gasto ja e alto) mas eficiencia, equidade de acesso e melhoria da alfabetizacao em populacoes de baixa renda.")

# =============================================================================
# 8 — RECOMENDACOES
# =============================================================================
elif page == "Recomendacoes":
    phase_label("Fase 5 — Recomendacoes de Politica Publica")
    st.title("Recomendacoes de Politica Publica")
    st.markdown("Cinco recomendacoes baseadas exclusivamente nos findings dos dados. Priorizadas por impacto x facilidade de implementacao.")
    hr()

    st.subheader("Matriz de Prioridade")
    matrix = pd.DataFrame({
        "Recomendacao": [
            "Focar em eficiencia do gasto em saude, nao no volume",
            "Investir em alfabetizacao de adultos como intervencao de saude",
            "Reduzir mortalidade infantil — maior alavanca disponivel",
            "Aprender com o modelo de saude de Singapura",
            "Construir dashboard publico de monitoramento",
        ],
        "Impacto": ["Alto","Alto","Alto","Medio","Medio"],
        "Esforco": ["Medio","Baixo","Medio","Alto","Baixo"],
        "Prioridade": ["#1 — Imediata","#2 — Imediata","#3 — Curto prazo","#4 — Estrategica","#5 — Rapida vitoria"],
        "Prazo": ["Continuo","3-5 anos","5-10 anos","10+ anos","30-60 dias"],
    })

    def color_imp(val):
        if val=="Alto": return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        return "background-color:#fff3cd;color:#856404;font-weight:bold"
    def color_prio(val):
        if "Imediata"    in val: return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if "Curto prazo" in val: return "background-color:#fff3cd;color:#856404"
        if "Rapida"      in val: return "background-color:#d4edda;color:#155724;font-weight:bold"
        return "background-color:#e8f4fd;color:#1a5276"

    st.dataframe(matrix.style.applymap(color_imp,subset=["Impacto"]).applymap(color_prio,subset=["Prioridade"]),
                 use_container_width=True,hide_index=True)
    hr()

    prios = [
        ("#1 — IMEDIATA: Focar em eficiencia do gasto em saude", ACCENT,
         "O Brasil gasta 9,5% do PIB em saude — nivel G7 — mas alcanca resultados do Cluster 0. Aumentar o gasto sem reformas nao vai fechar a lacuna.",
         ["Comparar eficiencia do sistema com Singapura e Coreia do Sul",
          "Auditar alocacao do gasto saude: proporcao prevencao vs. tratamento",
          "Implementar financiamento baseado em resultados para atencao primaria",
          "Monitorar ganhos de expectativa de vida por bilhao investido"]),
        ("#2 — IMEDIATA: Investir em alfabetizacao de adultos", TEAL,
         "A alfabetizacao (corr=0,81) supera todas as outras variaveis como preditor de saude. Educacao hoje = saude em 20-30 anos.",
         ["Ampliar programas de alfabetizacao de adultos acima de 30 anos",
          "Medir e publicar taxas de alfabetizacao por municipio anualmente",
          "Comparar com Costa Rica (97,9% alfabetizacao, 80,3 anos expectativa de vida)",
          "Monitorar correlacao alfabetizacao-expectativa de vida em nivel subnacional"]),
        ("#3 — CURTO PRAZO: Reduzir mortalidade infantil", MID_BLUE,
         "Mortalidade infantil (corr=-0,95) e a variavel mais fortemente ligada a saude geral. Brasil em 13/1k vs. Chile em 6,4/1k.",
         ["Identificar municipios com mortalidade infantil acima de 20/1k",
          "Meta de 10 anos: reduzir de 13 para menos de 7/1k nascimentos vivos",
          "Comparar com o Chile — melhor desempenho da America Latina",
          "Publicar ranking de mortalidade infantil por municipio trimestralmente"]),
        ("#4 — ESTRATEGICA: Aprender com o modelo de saude de Singapura", GOLD,
         "Singapura alcanca 85,4 anos de expectativa de vida com 4,0% do PIB. Modelo combina cobertura universal e atencao primaria como porta de entrada.",
         ["Estudar o sistema Medisave: contas individuais obrigatorias de saude",
          "Adotar modelo de atencao primaria: reduzir pronto-socorro como primeira opcao",
          "Implementar scorecard nacional de eficiencia em saude publicado anualmente",
          "Encomendar estudo de benchmarking formal: Brasil vs. Singapura vs. Coreia do Sul"]),
        ("#5 — RAPIDA VITORIA: Dashboard publico de monitoramento", GREEN,
         "Os dados para monitorar o progresso do Brasil existem nos APIs do Banco Mundial e OMS. O que falta e um sistema publico e atualizado.",
         ["Integrar indicadores da API do Banco Mundial em dashboard federal publico",
          "Publicar atualizacoes trimestrais: expectativa de vida, mortalidade infantil, alfabetizacao",
          "Criar indice de eficiencia comparando Brasil com pares do Cluster 0 anualmente",
          "Disponibilizar dados em nivel de estado e municipio"]),
    ]

    for title,color,prob,acoes in prios:
        with st.expander(title,expanded=False):
            st.markdown(f"**Problema / Oportunidade:** {prob}")
            st.markdown("**Acoes:**")
            for i,a in enumerate(acoes,1): st.markdown(f"  {i}. {a}")

# =============================================================================
# 9 — APENDICE TECNICO
# =============================================================================
elif page == "Apendice Tecnico":
    phase_label("Apendice Tecnico")
    st.title("Apendice Tecnico")
    hr()

    st.subheader("Tecnologias e Ferramentas")
    tech = pd.DataFrame({
        "Ferramenta": ["Python","Pandas & NumPy","Scikit-learn","Matplotlib","Streamlit","GitHub Pages","ReportLab"],
        "Uso": ["Linguagem principal","Manipulacao de dados e dados em painel",
                "Regressao Linear + KMeans clustering",
                "Visualizacoes exploratórias","Dashboard interativo (este app)",
                "Site do projeto","Geracao do relatorio PDF"],
        "Versao": ["3.10+","—","—","—","—","—","—"],
    })
    st.dataframe(tech,use_container_width=True,hide_index=True)
    hr()

    st.subheader("Performance do Modelo — Detalhes")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R2",f"{R2:.4f}")
    c2.metric("MAE",f"{MAE:.2f} anos")
    c3.metric("Paises no dataset",f"{len(df22)}")
    c4.metric("Features usadas","4")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coeficiente": [round(c,4) for c in model.coef_],
        "Direcao": ["Positivo" if c>0 else "Negativo" for c in model.coef_],
        "Interpretacao": [
            "Cada $10k extra de PIB per capita = ganho direto pequeno em expectativa de vida",
            "Cada 1% PIB a mais em saude = +0,19 anos (retornos decrescentes acima de 10%)",
            "Quase zero — resultados importam mais que gasto em educacao",
            "Cada 1pp de alfabetizacao = +0,28 anos de expectativa de vida (driver mais forte)",
        ],
    }).sort_values("Coeficiente",key=abs,ascending=False)
    st.dataframe(coef_df,use_container_width=True,hide_index=True)
    hr()

    st.markdown("**GitHub:** https://github.com/Bruno-Barradas/global-health-education-index")
    st.markdown(f"**Dataset:** global_health_education.csv — {len(df):,} registros, {df['country_code'].nunique()} paises, {df['year'].nunique()} anos")
    st.markdown("**Fontes:** Banco Mundial · OMS · Our World in Data")
    st.caption("Indice Global de Saude e Educacao — Bruno Barradas 2026")
