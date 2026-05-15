"""
app.py
Streamlit UI — wraps MentalHealthPipeline in an interactive web app.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import MentalHealthPipeline, PipelineResult


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Crisis sidebar (always visible) ──────────────────────────────────────────
st.sidebar.markdown("""
## 🆘 Crisis Support
If you or someone you know is in immediate danger:

| Resource | Contact |
|----------|---------|
| **988 Lifeline** (US) | Call or text **988** |
| **Crisis Text Line** | Text **HOME** to 741741 |
| **Emergency** | Call **911** |
| **International** | [IASP Directory](https://www.iasp.info/resources/Crisis_Centres/) |

---
""")

st.sidebar.markdown("""
## ℹ️ About
This tool uses a fine-tuned DistilBERT model to classify mental health
statements and surface relevant resources.

**Not a substitute for professional care.**
""")


# ── Pipeline loader (cached — loads once per session) ─────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_pipeline() -> MentalHealthPipeline:
    return MentalHealthPipeline(
        model_dir="models/distilbert_finetuned",
        data_dir="data",
        resources_path="mental_health_resources.json",
        top_k=5,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Support System")
st.caption(
    "Share what's on your mind. The system will identify patterns and "
    "suggest evidence-based coping resources."
)
st.warning(
    "⚠️ This tool is for informational purposes only and does not provide "
    "medical advice. Please consult a qualified healthcare provider.",
    icon="⚠️",
)

st.divider()


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="What's on your mind?",
    placeholder="Describe how you're feeling, what you're experiencing, or what's been troubling you…",
    height=160,
    help="The more detail you provide, the more accurate the analysis.",
)

analyze_btn = st.button("Analyze & Find Resources", type="primary", use_container_width=False)


# ── Result rendering ──────────────────────────────────────────────────────────

def _render_classification(result: PipelineResult):
    """Top row: predicted state + confidence + probability chart."""
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Classification")

        # Confidence colour: green ≥ 0.7, orange ≥ 0.5, red < 0.5
        conf_pct = result.confidence
        if conf_pct >= 0.70:
            badge = "🟢"
        elif conf_pct >= 0.50:
            badge = "🟡"
        else:
            badge = "🔴"

        st.metric(
            label="Predicted State",
            value=result.predicted_state,
            delta=f"{badge} {conf_pct:.1%} confidence",
        )

    with col_right:
        st.subheader("Probability Distribution")
        probs_df = (
            pd.DataFrame.from_dict(
                result.all_probabilities, orient="index", columns=["probability"]
            )
            .reset_index()
            .rename(columns={"index": "Category"})
            .sort_values("probability", ascending=True)
        )

        fig = px.bar(
            probs_df,
            x="probability",
            y="Category",
            orientation="h",
            color="probability",
            color_continuous_scale="Blues",
            range_x=[0, 1],
            labels={"probability": "Probability"},
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_resources(result: PipelineResult):
    """Resources section — organised by type."""
    st.subheader("💡 Recommended Resources")

    if not result.resources:
        st.info("No resources available for this category.")
        return

    # Separate crisis lines from general coping strategies
    crisis    = [r for r in result.resources if r.get("type") == "crisis_line"]
    general   = [r for r in result.resources if r.get("type") != "crisis_line"]

    # Crisis first if any
    if crisis:
        for r in crisis:
            with st.container(border=True):
                st.markdown(f"🚨 **{r['name']}**")
                st.write(r["description"])
                if "link" in r:
                    st.link_button("Open Resource", r["link"])

    # General resources in 2-column grid
    cols = st.columns(2)
    for i, r in enumerate(general):
        with cols[i % 2]:
            with st.container(border=True):
                icon = {"coping_strategy": "🛠️", "organization": "🏢", "resource": "📄"}.get(
                    r.get("type", ""), "📌"
                )
                st.markdown(f"{icon} **{r['name']}**")
                st.caption(f"Source: {r.get('source', 'N/A')}")
                st.write(r["description"])
                if "link" in r:
                    st.link_button("Learn More", r["link"])


def _render_similar(result: PipelineResult):
    """Similar statements section with category distribution chart."""
    st.subheader("👥 Semantically Similar Experiences")
    st.caption(
        "These are statements from the dataset that are most similar to yours. "
        "Lower distance = higher similarity."
    )

    if not result.similar_statements:
        st.info("No similar statements retrieved.")
        return

    col_list, col_chart = st.columns([2, 1])

    with col_list:
        for i, s in enumerate(result.similar_statements, 1):
            with st.expander(f"#{i} — {s.mental_state}  (dist: {s.distance:.2f})"):
                st.write(s.statement)

    with col_chart:
        st.markdown("**Category distribution of retrieved statements**")
        cat_counts = pd.Series(
            [s.mental_state for s in result.similar_statements]
        ).value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig2 = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
        st.plotly_chart(fig2, use_container_width=True)


# ── Main execution block ──────────────────────────────────────────────────────

if analyze_btn:
    text = user_input.strip()
    if not text:
        st.warning("Please enter some text before analyzing.")
    else:
        pipeline = load_pipeline()

        with st.spinner("Analyzing…"):
            result = pipeline.run(text)

        st.divider()
        _render_classification(result)
        st.divider()
        _render_resources(result)
        st.divider()
        _render_similar(result)
