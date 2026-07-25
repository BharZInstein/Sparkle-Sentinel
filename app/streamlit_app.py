import sys
import os
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.orchestrator import run_agent
from src.data_loader import load_dataset

st.set_page_config(page_title="AML Suspicious Activity Agent", layout="wide")
st.title("AI-Powered Suspicious Activity Detection Agent")

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Laundering rate: {df['Is_laundering'].mean()*100:.3f}%")

query = st.text_input("Ask the agent", placeholder="e.g. Find structuring patterns in the last 30 days")
sample_size = st.sidebar.slider("Sample size for demo speed", 1000, 50000, 20000, step=1000)

if st.button("Run Agent") and query:
    with st.spinner("Agent parsing intent and executing plan..."):
        sample_df = df.sample(sample_size, random_state=42)
        result = run_agent(query, sample_df)

    st.subheader("Execution Summary")
    st.write(f"**Query:** {result['query']}")
    st.write(f"**Detected intent:** {result['detected_intent']}")
    st.write(f"**Detected pattern:** {result['detected_pattern']}")
    st.write(f"**Tools invoked:** {' → '.join(result['tools_invoked'])}")

    if result.get("eda_summary"):
        st.subheader("EDA Summary")
        st.json(result["eda_summary"])
        if result["eda_summary"].get("payment_type_dist"):
            fig = px.bar(
                x=list(result["eda_summary"]["payment_type_dist"].keys()),
                y=list(result["eda_summary"]["payment_type_dist"].values()),
                labels={"x": "Payment Type", "y": "Proportion"},
                title="Payment Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    if result.get("aggregation_result"):
        st.subheader("Aggregation Result")
        st.dataframe(pd.DataFrame(result["aggregation_result"]))

    if result.get("flags"):
        st.subheader(f"Flagged Items ({result['flag_count']} total, {result['high_risk_count']} high-risk)")
        st.dataframe(pd.DataFrame(result["flags"]))
        for f in result["flags"]:
            with st.expander(f"{f['Sender_account']} → {f['Receiver_account']} | {f['risk_level']}"):
                st.write(f"**Amount:** {f['Amount']}")
                st.write(f"**Recommended action:** {f['recommended_action']}")
                st.write(f"**Explanation:** {f['explanation']}")