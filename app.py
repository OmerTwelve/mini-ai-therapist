import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import MentalHealthPipeline, PipelineResult


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Therapist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Crisis sidebar (always visible) ──────────────────────────────────────────

st.sidebar.markdown("""
## ℹ️ About
This AI-therapist prototype uses a fine-tuned DistilBERT model to detect
mental-health patterns, generate supportive reflections, and surface relevant resources.

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
st.title("AI Therapist")
st.caption(
    "Share what's on your mind. The assistant will reflect back what it hears, "
    "offer a grounding prompt, and suggest support resources."
)
st.warning(
    "This educational tool is not a licensed therapist and does not provide "
    "medical advice. Please consult a qualified healthcare provider.",
    icon="⚠️",
)

st.divider()


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    label="What would you like to talk through?",
    placeholder="Write about what happened, what you are feeling, or what has been hard lately...",
    height=160,
    help="The more detail you provide, the more useful the reflection can be.",
)

analyze_btn = st.button("Respond", type="primary", use_container_width=False)


# ── Result rendering ──────────────────────────────────────────────────────────

def _render_therapist_response(result: PipelineResult):
    """Therapist-style response shown before diagnostics."""
    st.subheader("Response")
    st.markdown(result.therapist_reply)

    if result.predicted_state == "Suicidal":
        st.error(
            "If there is immediate danger, call emergency services now. In the US, call or text 988 for crisis support.",
            icon="🚨",
        )

    st.markdown("**A question to sit with:**")
    st.info(result.reflection_prompt)


def _render_classification(result: PipelineResult):
    """Top row: predicted state + confidence + probability chart."""
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Clinical Signal")

        # Confidence colour: green ≥ 0.7, orange ≥ 0.5, red < 0.5
        conf_pct = result.confidence
        if conf_pct >= 0.70:
            badge = "🟢"
        elif conf_pct >= 0.50:
            badge = "🟡"
        else:
            badge = "🔴"

        st.metric(
            label="Detected Pattern",
            value=result.predicted_state,
            delta=f"{badge} {conf_pct:.1%} confidence",
        )

    with col_right:
        st.subheader("Model Confidence")
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
    st.subheader("Support Tools")

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
    st.subheader("Related Experiences")
    st.caption(
        "These are anonymized dataset statements with similar language patterns. "
        "Lower distance = higher similarity."
    )

    if not result.similar_statements:
        st.info("No similar statements retrieved.")
        return

    col_list, col_chart = st.columns([2, 1])

    with col_list:
        for i, s in enumerate(result.similar_statements, 1):
            with st.expander(f"#{i} — {s.mental_state}  (distance: {s.distance:.2f})"):
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

        with st.spinner("Thinking through your message..."):
            result = pipeline.run(text)

        st.divider()
        _render_therapist_response(result)
        st.divider()
        _render_resources(result)
        st.divider()
        with st.expander("Show model details"):
            _render_classification(result)
            st.divider()
            _render_similar(result)
