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
