# Angle 3 — Développement

from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

mydf = load_mydf()  

FIG_DIR = "figures/003"

def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")

# Préparation de la variable éco balance

df3 = mydf.copy()
df3["Eco_Balance"] = df3["Total_Biocapacity"] - df3["Total_Footprint_Consumption"]

# HDI vs Footprint Consumption

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="HDI", y="Total_Footprint_Consumption",
    color="darkred", s=70
)

# lowess (regression douce)
low = lowess(df3["Total_Footprint_Consumption"], df3["HDI"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.title("HDI vs Empreinte de Consommation")
plt.xlabel("HDI")
plt.ylabel("Empreinte écologique (consommation)")

# annoter quelques pays extrêmes
top = df3.sort_values("Total_Footprint_Consumption", ascending=False).head(3)
bottom = df3.sort_values("Total_Footprint_Consumption").head(3)

for _, r in pd.concat([top, bottom]).iterrows():
    plt.annotate(r["Country"], (r["HDI"], r["Total_Footprint_Consumption"]),
                 textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_HDI_vs_Footprint")
plt.show()


# GDP vs Footprint Consumption

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="Per Capita GDP", y="Total_Footprint_Consumption",
    color="steelblue", s=70
)

low = lowess(df3["Total_Footprint_Consumption"], df3["Per Capita GDP"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.title("PIB par habitant vs Empreinte de Consommation")
plt.xlabel("PIB par habitant")
plt.ylabel("Empreinte écologique (consommation)")

for _, r in pd.concat([top, bottom]).iterrows():
    plt.annotate(r["Country"], (r["Per Capita GDP"], r["Total_Footprint_Consumption"]),
        textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_GDP_vs_Footprint")
plt.show()


# HDI vs Ecological Deficit

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="HDI", y="Eco_Balance",
    color="forestgreen", s=70
)

low = lowess(df3["Eco_Balance"], df3["HDI"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.axhline(0, color="gray", linestyle="--")  # ligne d'équilibre

plt.title("HDI vs Équilibre Écologique (Biocapacité – Empreinte)")
plt.xlabel("HDI")
plt.ylabel("Équilibre écologique")

# annoter pays en déficit extrême
ext = df3.sort_values("Eco_Balance").head(3)
for _, r in ext.iterrows():
    plt.annotate(r["Country"], (r["HDI"], r["Eco_Balance"]),
        textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_HDI_vs_EcoBalance")
plt.show()


# On note les pays un peu paradoxaux
#   (haut HDI mais faible empreinte)

paradox = df3[
    (df3["HDI"] >= 0.75) &
    (df3["Total_Footprint_Consumption"] <= 2.5) &
    (df3["Eco_Balance"] >= -1)
][["Country", "HDI", "Total_Footprint_Consumption", "Eco_Balance"]]

print("\n PAYS PARADOXAUX (HDI élevé, faible empreinte) ")
print(paradox)
