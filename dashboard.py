import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(layout="wide")
st.title("Modèle fitting interactif")

# ======================================================
# CHARGEMENT DES DONNÉES
# ======================================================

@st.cache_data
def load_mydf():
    df = pd.read_excel("data/toutlespays.xlsx")

    rename_dict = {
        "actual \nCountry Overshoot Day \n2018": "Overshoot Day",
        "Cropland Footprint": "Cropland_Footprint_Production",
        "Grazing Footprint": "Grazing_Footprint_Production",
        "Forest Product Footprint": "Forest_Footprint_Production",
        "Fish Footprint": "Fish_Footprint_Production",
        "Built up land": "BuiltUp_Footprint_Production",
        "Carbon Footprint": "Carbon_Footprint_Production",
        "Cropland Footprint.1": "Cropland_Footprint_Consumption",
        "Grazing Footprint.1": "Grazing_Footprint_Consumption",
        "Forest Product Footprint.1": "Forest_Footprint_Consumption",
        "Fish Footprint.1": "Fish_Footprint_Consumption",
        "Built up land.1": "BuiltUp_Footprint_Consumption",
        "Carbon Footprint.1": "Carbon_Footprint_Consumption",
        "Built up land.2": "BuiltUp_Biocapacity",
        "Total biocapacity ": "Total_Biocapacity",
        "Total Ecological Footprint (Production)": "Total_Footprint_Production",
        "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
    }

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

    force_text = ["Country", "Region", "Income Group", "Overshoot Day", "Quality Score"]

    def clean_numeric(s):
        if s.dtype != object:
            return s
        s = s.astype(str)
        s = s.replace(["-", "--", "", "…"], np.nan)
        s = s.str.replace(r"[,$% ]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    for col in df.columns:
        if col not in force_text:
            df[col] = clean_numeric(df[col])

    parsed = pd.to_datetime(df["Overshoot Day"], errors="coerce")
    df = df.loc[~parsed.isna()].copy()
    df["Overshoot_Day_DOY"] = parsed.dt.dayofyear

    df["Income Group"] = pd.Categorical(
        df["Income Group"],
        categories=["LI", "LM", "UM", "HI"],
        ordered=True
    )

    df = df.drop(columns=["Country", "Region", "Overshoot Day", "Quality Score"])
    return df.dropna()


df = load_mydf()
train_df, _ = train_test_split(df, test_size=0.2, random_state=123)

# ======================================================
# GROUPES DE VARIABLES
# ======================================================

VAR_GROUPS = {
    "Socio-économique": [
        "SDGi",
        'Q("Life Expectancy")',
        "HDI",
        'Q("Per Capita GDP")',
        'Q("Population (millions)")',
        'C(Q("Income Group"), Treatment(reference="LI"))',
    ],
    "Production": [
        "Cropland_Footprint_Production",
        "Grazing_Footprint_Production",
        "Forest_Footprint_Production",
        "Fish_Footprint_Production",
        "BuiltUp_Footprint_Production",
        "Carbon_Footprint_Production",
    ],
    "Consommation": [
        "Cropland_Footprint_Consumption",
        "Grazing_Footprint_Consumption",
        "Forest_Footprint_Consumption",
        "Fish_Footprint_Consumption",
        "BuiltUp_Footprint_Consumption",
        "Carbon_Footprint_Consumption",
    ],
    "Biocapacité": [
        "Cropland",
        'Q("Grazing land")',
        'Q("Forest land")',
        'Q("Fishing ground")',
        "BuiltUp_Biocapacity",
        "Total_Biocapacity",
    ],
    "Dérivées": [
        'Q("Ecological (Deficit) or Reserve")',
        'Q("Number of Earths required")',
        'Q("Number of Countries required")',
    ],
    "Totaux": [
        "Total_Footprint_Production",
        "Total_Footprint_Consumption",
    ],
}

ALL_VARS = [v for group in VAR_GROUPS.values() for v in group]

# ======================================================
# SIDEBAR – SÉLECTION
# ======================================================

for group, vars in VAR_GROUPS.items():
    if group not in st.session_state:
        st.session_state[group] = []


st.sidebar.header("Spécification du modèle")

select_all = st.sidebar.checkbox(
    "Tout sélectionner",
    key="select_all"
)

selected_terms = []

for group, vars in VAR_GROUPS.items():
    # Si "tout sélectionner" est coché, on force la sélection
    if select_all:
        st.session_state[group] = vars.copy()

    with st.sidebar.expander(group):
        picks = st.multiselect(
            "Variables",
            vars,
            key=group
        )
        selected_terms.extend(picks)


# ======================================================
# FORMULE MODIFIABLE
# ======================================================

st.subheader("Formule du modèle")

default_formula = (
    "Overshoot_Day_DOY ~ " + " + ".join(selected_terms)
    if selected_terms else
    "Overshoot_Day_DOY ~ 1"
)

formula_user = st.text_area(
    "Modifier :",
    value=default_formula,
    height=100
)

# ======================================================
# BOUTON LANCER
# ======================================================

if not st.button("Lancer le modèle"):
    st.stop()

# ======================================================
# ESTIMATION INITIALE (SILENCIEUSE)
# ======================================================

model_ols = smf.ols(formula_user, data=train_df).fit()

# ======================================================
# TEST D’HOMOSCÉDASTICITÉ (BREUSCH–PAGAN)
# ======================================================

bp_stat, bp_pvalue, _, _ = het_breuschpagan(
    model_ols.resid,
    model_ols.model.exog
)

st.subheader("Test d’homoscédasticité (Breusch–Pagan)")

st.write(f"Statistique LM : {bp_stat:.3f}")
st.write(f"P-valeur : {bp_pvalue:.4f}")

# ======================================================
# DÉCISION AUTOMATIQUE : ROBUSTE OU NON
# ======================================================

if bp_pvalue < 0.05:
    st.warning(
        "Hétéroscédasticité détectée (p < 0.05) → "
        "ré-estimation avec erreurs standards robustes (HC1)."
    )
    model_final = smf.ols(
        formula_user,
        data=train_df
    ).fit(cov_type="HC1")
    robust_label = "Erreurs standards robustes (HC1)"
else:
    st.success(
        "Homoscédasticité non rejetée (p ≥ 0.05) → "
        "modèle OLS classique conservé."
    )
    model_final = model_ols
    robust_label = "Erreurs standards classiques"

st.subheader(f"Résumé du modèle ({robust_label})")
st.code(model_final.summary().as_text(), language="text")

st.subheader("Résidus vs valeurs ajustées")

fig, ax = plt.subplots(figsize=(4.5, 3.2))  # plus petit

ax.scatter(
    model_final.fittedvalues,
    model_final.resid,
    color="black",      # points noirs
    alpha=0.7,
    s=25                # taille des points
)

ax.axhline(0, color="red", linewidth=1)

ax.set_xlabel("Valeurs ajustées")
ax.set_ylabel("Résidus")

ax.set_title("Résidus vs valeurs ajustées", fontsize=10)

plt.tight_layout()
st.pyplot(fig, use_container_width=False)




st.subheader("Diagnostics de la matrice de conception")

_, X = patsy.dmatrices(formula_user, train_df, return_type="dataframe")

svals = np.linalg.svd(X, compute_uv=False)
tol = np.finfo(float).eps * max(X.shape) * svals[0]
null_mask = svals < tol

col1, col2 = st.columns(2)

with col1:
    st.metric("Nombre de colonnes", X.shape[1])
    st.metric("Rang de la matrice", np.linalg.matrix_rank(X))

with col2:
    if null_mask.any():
        st.error("Matrice non de plein rang")
    else:
        st.success("Matrice de plein rang")

if null_mask.any():
    st.markdown("### Dépendances linéaires exactes")
    _, _, Vt = svd(X, full_matrices=False)
    for i, v in enumerate(Vt[null_mask]):
        st.write(f"Dépendance {i+1}")
        st.dataframe(pd.Series(v, index=X.columns))
