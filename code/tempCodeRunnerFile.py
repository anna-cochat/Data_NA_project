from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

mydf = load_mydf()  
print(mydf.columns)

FIG_DIR = "figures/004"

def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")


# On calcule l'écart
mydf['Footprint_Gap'] = mydf['Total_Footprint_Consumption'] - mydf['Total_Footprint_Production']

plt.figure(figsize=(16,6))
ordered = mydf.sort_values('Footprint_Gap')

plt.bar(ordered['Country'], ordered['Footprint_Gap'],
        color=['darkgreen' if x >= 0 else 'darkred' for x in ordered['Footprint_Gap']])

plt.xticks(rotation=90)
plt.axhline(0, color='black', linewidth=1)
plt.title("Gap Empreinte : Consommation - Production (positif = externalisation)")
plt.ylabel("Gap (gha/personne)")
plt.tight_layout()

savefig("004_GapBarplot")
plt.show()


# Production vs consommation
plt.figure(figsize=(8,6))
plt.scatter(mydf['Total_Footprint_Production'], mydf['Total_Footprint_Consumption'], s=80)

max_val = max(mydf['Total_Footprint_Production'].max(), mydf['Total_Footprint_Consumption'].max())
plt.plot([0, max_val], [0, max_val], '--', color='gray')

plt.xlabel("Empreinte de Production")
plt.ylabel("Empreinte de Consommation")
plt.title("Production vs Consommation")

plt.tight_layout()
savefig("004_Production_vs_Consumption")
plt.show()


# top 10 externaliseurs
top_ext = mydf.sort_values('Footprint_Gap', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.barh(top_ext['Country'], top_ext['Footprint_Gap'], color='darkgreen')
plt.gca().invert_yaxis()

plt.xlabel("Gap (Consommation - Production)")
plt.title("Top externalisateurs d'empreinte écologique")
plt.tight_layout()

savefig("004_TopExternalizers")
plt.show()

# ===============================================================
# ANGLE 5 — Regional Profiles
# ===============================================================

print("\n=== Angle 5: Regional Profiles ===\n")

# --- VARIABLES FOR REGION ANALYSIS ---
key_vars = [
    "HDI",
    "Per Capita GDP",
    "Total_Footprint_Consumption",
    "Total_Biocapacity"
]

components_prod = [
    "Cropland_Footprint_Production",
    "Grazing_Footprint_Production",
    "Forest_Footprint_Production",
    "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production",
    "Carbon_Footprint_Production"
]

# --- Force conversion of footprint components to numeric ---
for col in components_prod:
    mydf[col] = pd.to_numeric(mydf[col], errors="coerce")


# --- MELT ---
mydf_melt = mydf.melt(
    id_vars="Region",
    value_vars=key_vars,
    var_name="Indicator",
    value_name="Value"
)

# --- BOX PLOT ---
plt.figure(figsize=(14,8))
sns.boxplot(data=mydf_melt, x="Indicator", y="Value", hue="Region")
plt.xticks(rotation=45)
plt.title("Distribution des indicateurs par région")
plt.tight_layout()
plt.savefig("../figures/005/005_Boxplots_Regions.png")
print("Saved: 005_Boxplots_Regions")
plt.close()

# --- STRIPPLOT ---
plt.figure(figsize=(14,8))
sns.stripplot(
    data=mydf_melt,
    x="Indicator",
    y="Value",
    hue="Region",
    dodge=True,
    alpha=0.7
)
plt.xticks(rotation=45)
plt.title("Répartition individuelle par région")
plt.tight_layout()
plt.savefig("../figures/005/005_Stripplots_Regions.png")
print("Saved: 005_Stripplots_Regions")
plt.close()

# --- REGIONAL AVERAGES OF FOOTPRINT PRODUCTION ---
regional_avgs = mydf.groupby("Region")[components_prod].mean()

plt.figure(figsize=(16,8))
regional_avgs.plot(kind="bar")
plt.title("Moyenne régionale des composantes de l'empreinte (production)")
plt.ylabel("gha/personne")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/005/005_Regional_Component_Averages.png")
print("Saved: 005_Regional_Component_Averages")
plt.close()

print("=== Angle 5 completed successfully ===")
