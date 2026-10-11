import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa

st.set_page_config(layout="wide")
st.title("🧫 Microbiome Diversity Dashboard")

# Sidebar for uploads
st.sidebar.header("Upload Your Files")
otu_file = st.sidebar.file_uploader("OTU Table (samples x OTUs)", type="csv")
metadata_file = st.sidebar.file_uploader("Metadata File", type="csv")
taxonomy_file = st.sidebar.file_uploader("Taxonomy File (optional)", type="csv")

if otu_file and metadata_file:
    otu_df = pd.read_csv(otu_file, index_col=0)
    meta_df = pd.read_csv(metadata_file, index_col=0)

    # Sync samples between OTU and metadata
    common_samples = otu_df.index.intersection(meta_df.index)
    otu_df = otu_df.loc[common_samples]
    meta_df = meta_df.loc[common_samples]

    group_col = meta_df.columns[0]
    st.sidebar.success(f"Found {len(common_samples)} matching samples.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Alpha Diversity", 
        "Beta Diversity", 
        "Taxonomic Composition", 
        "Differential Abundance"
    ])

    with tab1:
        st.subheader("🧬 Alpha Diversity")

        metric = st.selectbox("Select Metric", ["shannon", "simpson", "chao1"])
        alpha_vals = alpha_diversity(metric=metric, counts=otu_df.values, ids=otu_df.index)
        alpha_df = pd.DataFrame({metric: alpha_vals, "Group": meta_df[group_col].values})

        fig1 = plt.figure(figsize=(8, 4))
        sns.boxplot(x="Group", y=metric, data=alpha_df)
        plt.title(f"{metric.title()} Diversity")
        st.pyplot(fig1)

        st.download_button("Download Alpha Diversity CSV", 
                           alpha_df.to_csv().encode(), 
                           file_name="alpha_diversity.csv")

    with tab2:
        st.subheader("📏 Beta Diversity (Bray-Curtis + PCoA)")

        beta = beta_diversity("braycurtis", otu_df.values, ids=otu_df.index)
        pcoa_results = pcoa(beta)
        coords = pcoa_results.samples.copy()
        coords["Group"] = meta_df[group_col].values

        fig2 = px.scatter(coords, x="PC1", y="PC2", color="Group",
                          title="PCoA Plot (Bray-Curtis)",
                          labels={"PC1": "PC1", "PC2": "PC2"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("🧬 Taxonomic Composition")

        if taxonomy_file:
            tax_df = pd.read_csv(taxonomy_file, index_col=0)
            tax_level = st.selectbox("Select taxonomic level", tax_df.columns.tolist())

            merged = otu_df.dot(pd.get_dummies(tax_df[tax_level]))
            rel_abundance = merged.div(merged.sum(axis=1), axis=0)

            top_taxa = rel_abundance.mean().sort_values(ascending=False).head(10).index
            filtered = rel_abundance[top_taxa]

            fig3, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(filtered, cmap="viridis", ax=ax)
            plt.title("Relative Abundance Heatmap")
            st.pyplot(fig3)

            st.download_button("Download Relative Abundance CSV", 
                               filtered.to_csv().encode(), 
                               file_name="relative_abundance.csv")
        else:
            st.warning("Please upload a taxonomy file to view taxonomic composition.")

    with tab4:
        st.subheader("📉 Differential Abundance (Log2 Fold Change)")

        groups = meta_df[group_col].unique()
        if len(groups) != 2:
            st.warning("Differential abundance requires exactly 2 groups.")
        else:
            g1, g2 = groups
            g1_samples = meta_df[meta_df[group_col] == g1].index
            g2_samples = meta_df[meta_df[group_col] == g2].index

            g1_mean = otu_df.loc[g1_samples].mean()
            g2_mean = otu_df.loc[g2_samples].mean()

            log2fc = np.log2(g2_mean + 1) - np.log2(g1_mean + 1)
            log2fc_df = pd.DataFrame({
                "OTU": log2fc.index,
                "Log2 Fold Change": log2fc.values
            }).sort_values(by="Log2 Fold Change", ascending=False)

            fig4 = px.bar(log2fc_df.head(20), x="OTU", y="Log2 Fold Change",
                          title=f"Top 20 OTUs (log2FC: {g2} vs {g1})")
            st.plotly_chart(fig4)

            st.download_button("Download Log2 Fold Change CSV", 
                               log2fc_df.to_csv(index=False).encode(), 
                               file_name="log2_fold_change.csv")

else:
    st.info("Upload both an OTU table and metadata file to get started.")
