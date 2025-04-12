import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# === Step 1: Load and Clean Data ===
leases = pd.read_csv(r"C:\Users\sammi\Downloads\OneDrive_2025-04-11\Dataset 2025\Leases.csv")

# Standardize market names
leases["market"] = leases["market"].replace({
    "Dallas/Fort Worth": "Dallas",
    "South Bay / San Jose": "San Jose"
})

# Create CBD flag
leases["cbd_flag"] = leases["CBD_suburban"].apply(lambda x: 1 if x == "CBD" else 0)

# === Step 2: Aggregate Market-Level Metrics (2022–2024) ===
leases_filtered = leases[(leases["year"] >= 2022) &
                         leases["overall_rent"].notna() &
                         leases["availability_proportion"].notna()]

market_summary = leases_filtered.groupby("market").agg({
    "overall_rent": "mean",
    "availability_proportion": "mean",
    "cbd_flag": "mean"
}).rename(columns={
    "overall_rent": "avg_rent",
    "availability_proportion": "avg_avail_prop",
    "cbd_flag": "cbd_ratio"
}).dropna()

# === Step 3: Normalize the Data ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(market_summary)
scaled_df = pd.DataFrame(scaled_data, index=market_summary.index, columns=market_summary.columns)

# === Step 4: Distance Matrix for Similarity ===
distance_matrix = pdist(scaled_df, metric='euclidean')
distance_matrix_square = squareform(distance_matrix)

# === Step 5: Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix_square, xticklabels=scaled_df.index, yticklabels=scaled_df.index, cmap='viridis')
plt.title("City-to-City Similarity Heatmap")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("heatmap_similarity.png")  
plt.show()
