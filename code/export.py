# export_to_r.py
from nettoyage import prepare_data

df_imputation, _ = prepare_data()

# IMPORTANT: keep categorical variables as they are
df_imputation.to_csv("data_for_mice.csv")
