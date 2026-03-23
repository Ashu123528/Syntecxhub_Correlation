import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Correlation Analysis", layout="wide")
st.title("📊 Correlation Heatmap & Pairwise Relationships")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# -------------------------------
# SELECT NUMERIC COLUMNS
# -------------------------------
cols = [
    'SalePrice', 
    'GrLivArea', 
    'OverallQual', 
    'GarageArea', 
    'TotalBsmtSF'
]

df = df[cols]

# -------------------------------
# CLEANING
# -------------------------------
df = df.fillna(df.mean(numeric_only=True))
df = df.drop_duplicates()

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# HEATMAP (WITH MASK)
# -------------------------------
st.subheader("🔥 Correlation Heatmap")

corr = df.corr(method='pearson')

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

heatmap_path = os.path.join(OUTPUT_DIR, "heatmap.png")
fig.savefig(heatmap_path, bbox_inches='tight')

st.pyplot(fig)

# -------------------------------
# PAIRPLOT
# -------------------------------
st.subheader("📉 Pairwise Relationships (Pairplot)")

pairplot = sns.pairplot(df)
pairplot_path = os.path.join(OUTPUT_DIR, "pairplot.png")
pairplot.savefig(pairplot_path)

st.pyplot(pairplot)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Correlation Insights")

corr_values = corr.where(~np.eye(corr.shape[0], dtype=bool))

max_corr = corr_values.max().max()
max_pair = corr_values.stack().idxmax()

min_corr = corr_values.min().min()
min_pair = corr_values.stack().idxmin()

insights = []

insights.append(f"Strongest Positive Correlation: {max_pair} -> {max_corr:.2f}")
insights.append(f"Strongest Negative Correlation: {min_pair} -> {min_corr:.2f}")

st.write(insights[0])
st.write(insights[1])

# -------------------------------
# SAVE INSIGHTS (UTF-8 FIX)
# -------------------------------
insight_path = os.path.join(OUTPUT_DIR, "insights.txt")

with open(insight_path, "w", encoding="utf-8") as f:
    for line in insights:
        f.write(line + "\n")

# -------------------------------
# DOWNLOAD BUTTONS
# -------------------------------
st.markdown("### 📥 Download Results")

with open(heatmap_path, "rb") as f:
    st.download_button("Download Heatmap", f, file_name="heatmap.png")

with open(pairplot_path, "rb") as f:
    st.download_button("Download Pairplot", f, file_name="pairplot.png")

with open(insight_path, "rb") as f:
    st.download_button("Download Insights", f, file_name="insights.txt")

# -------------------------------
# END
# -------------------------------
st.markdown("---")
st.write("✅ Correlation Analysis Completed Successfully")