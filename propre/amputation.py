import numpy as np
import pandas as pd
from scipy import optimize

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

import missingno

from nettoyage import prepare_data

np.random.seed(42)
df_imputation, _, _ = prepare_data()

df_imputation_complete = df_imputation.dropna()
df_amputation = df_imputation_complete.drop(columns=[
    "Income Group", "Income_Group_Code",
    "Total_Footprint_Production", "Total_Footprint_Consumption",
    "Total_Biocapacity", "Quality Score", "Population (millions)",
    "Region", "Quality_Score_Code", "Overshoot_Day_DOY"
], errors="ignore")

df_amputation_na = df_imputation.drop(columns=[
    "Income Group", "Income_Group_Code", 
    "Total_Footprint_Production", "Total_Footprint_Consumption",
    "Total_Biocapacity", "Quality Score", "Population (millions)",
    "Region", "Quality_Score_Code", "Overshoot_Day_DOY"
], errors="ignore")

missingno.dendrogram(df_amputation_na)
plt.show()

cols_footprints = [
    'Grazing_Footprint_Production', 'Cropland_Footprint_Production',
    'Forest_Footprint_Production', 'Fish_Footprint_Production',
    'BuiltUp_Footprint_Production', 'Carbon_Footprint_Production',
    'Cropland_Footprint_Consumption', 'Grazing_Footprint_Consumption',
    'Forest_Footprint_Consumption', 'Fish_Footprint_Consumption', 'Carbon_Footprint_Consumption',   
    'Grazing land', 'Forest land', 'Fishing ground'
]

cols_inde = ['Life Expectancy', 'SDGi']
cols_gdp_hdi = ['Per Capita GDP', 'HDI']

p_footprints = df_amputation_na[cols_footprints].isnull().mean().mean()
p_gdp  = df_amputation_na['Per Capita GDP'].isnull().mean()
p_hdi  = df_amputation_na['HDI'].isnull().mean()
p_life = df_amputation_na['Life Expectancy'].isnull().mean()
p_sdgi = df_amputation_na['SDGi'].isnull().mean()
p_gdp_hdi = (p_gdp + p_hdi) / 2

print("p footprints : ", p_footprints)
print("p GDP : ", p_gdp)
print("p HDI : ", p_hdi)
print("p Life Expectancy : ", p_life)
print("p SDGi : ", p_sdgi)
print("p GDP_HDI : ", p_gdp_hdi)

n = len(df_amputation_na)
xtrou = df_amputation_na.copy()

np.random.seed(42)

miss_id = np.random.uniform(0, 1, size=n) < p_footprints
for col in cols_footprints:
    xtrou.loc[miss_id, col] = np.nan

miss_id = np.random.uniform(0, 1, size=n) < p_gdp_hdi
for col in cols_gdp_hdi:
    xtrou.loc[miss_id, col] = np.nan

for col, p in zip(cols_inde, [p_life, p_sdgi]):
    miss_id = np.random.uniform(0, 1, size=n) < p
    xtrou.loc[miss_id, col] = np.nan

missingno.heatmap(df_amputation_na)
plt.show()
missingno.heatmap(xtrou)
plt.show()

missingno.matrix(df_amputation_na)
plt.show()
missingno.matrix(xtrou)
plt.show()
print(len(df_amputation_na), "  ", len(xtrou))

df_amputed = df_imputation.copy()
df_amputed[xtrou.columns] = xtrou

print(df_amputed.head(5))
missingno.matrix(df_amputed)
plt.show()

df_imputation.to_csv("df_avant_amputation.csv")
df_amputed.to_csv("data/df_amputed.csv")