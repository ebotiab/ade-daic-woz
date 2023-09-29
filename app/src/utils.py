import plotly.express as px
import streamlit as st


def statistics_pd_col(pd_col, display=True):
    stats = pd_col.min(), pd_col.max(), pd_col.mean(), pd_col.std()
    if display:
        stats_labels = ["Min", "Max", "Mean", "Std"]
        [c.write(f"**{lab}**: {round(s, 2)}") for c, lab, s in zip(st.columns(4), stats_labels, stats)]
    return stats


def display_distribution(df, x_var, y_vars, st_container=None, title=""):
    fig = px.bar(df, x=df.index if x_var == "index" else x_var, y=y_vars, title=title)
    st_container = st_container if st_container else st
    st_container.plotly_chart(fig, use_container_width=True)
