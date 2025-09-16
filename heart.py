import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Heart Disease EDA Dashboard", layout="wide")

st.title("❤️ Heart Disease EDA Dashboard")

# Load data
df = pd.read_csv("heart.csv")

# Sidebar
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show Raw Data")

# Show data
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

# Dataset info
st.subheader("Dataset Info")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))

# Missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Summary stats
st.subheader("Summary Statistics")
st.write(df.describe().T)

# Distribution plots
st.subheader("Distributions of Numerical Features")
fig, ax = plt.subplots(figsize=(12, 10))
df.hist(ax=ax, bins=20)
st.pyplot(fig)

# Boxplots
st.subheader("Boxplots of Numerical Features")
num_cols = df.select_dtypes(include=np.number).columns
fig, axes = plt.subplots(len(num_cols)//3 + 1, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Countplots (compact grid version)
st.subheader("Categorical Feature Distributions")
cat_cols = [col for col in df.columns if df[col].nunique() < 10]
fig, axes = plt.subplots((len(cat_cols)+2)//3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=df, hue="target", ax=axes[i])
    axes[i].set_title(col)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("📊 Insights")
st.markdown("""
- Dataset has **no missing values**.  
- Features like **cp, thalach, oldpeak** strongly correlate with target.  
- Some features are categorical (`sex, cp, fbs, restecg, exang, slope, ca, thal`).  
- Target distribution is fairly balanced.  
""")
