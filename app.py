"""Streamlit web interface for the LLM Personality stability experimental procedure workflow.
This module defines a Streamlit app that allows users to configure and run the personality stability experiment on multiple model instances. It provides a user-friendly interface for inputting experimental parameters, launching the experiment, and visualizing results.
The app includes:
- A form for configuring group attributes (names, initial personalities, number of instances)
- A form for configuring nudge attributes (direction, number of nudges or destination)
- A form for selecting the personality test and analysis options
- A cached function to build the experimental workflow graph (runs once per session)
- A function to create the initial state for each model instance based on user input
- A main section that runs the experiment when the user clicks the button, handles errors gracefully,
and displays the results in a dataframe and optional visualizations.


"""

import streamlit as st
import ast
from dotenv import load_dotenv

load_dotenv()

from ai_in_loop.config import Config
from ai_in_loop.graph import build_graph
from ai_in_loop.state import ModelInstanceState
from ai_in_loop.experiment import run_experiment, summarize_experiment, plot_trajectories, prepare_analysis_df



# ── Launch Command ───────────────────────────────────────────────────

# streamlit run app.py

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="AI Personality Experiment", page_icon="📄", layout="wide")
st.title("AI Personality Experiment")


# ── Cached graph + helper ────────────────────────────────────────────
# @st.cache_resource is like st.session_state but shared across all
# browser tabs and only runs the function once. It's the recommended
# way to cache expensive objects like compiled graphs or DB connections.
@st.cache_resource
def get_pipeline():
    """Build and cache the LangGraph workflow (runs once per session)."""
    cfg = Config.from_env()
    return build_graph(cfg)


def get_initial_state(personality_test: str, initialization_attempt_limit: int, nudge_direction: str, initial_personality = None, nudge_num_limit = None, nudge_destination = None,) -> ModelInstanceState:
    """Create the initial workflow state from a URL."""
    return {
        "llm_ID": 0,
        "message_history": [],
        "personality_test": personality_test,
        "measurements": [],
        "responses": [],
        "initial_personality": initial_personality,
        "current_personality": None,
        "initialization_attempts": 0,
        "initialization_attempt_limit": initialization_attempt_limit,
        "successfully_initialized": None,
        "nudges_performed": 0,
        "nudge_num_limit": nudge_num_limit,
        "nudge_destination": nudge_destination,
        "nudge_destination_reached": None,
        "nudge_direction": nudge_direction,
        "warnings": [],
    }


# Human-readable labels for each workflow node
NODE_LABELS = {
    "initialize personality": "Initializing personality...",
    "pre_exp_measure": "Measuring personality...",
    "nudge": "Performing nudge...",
    "post_exp_measure": "Measuring personality...",
}


col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Group Attributes")
    group_names = st.text_input("Group Names")
    group_personalities = st.text_input("Group Initial Personalities")
    attempt_limit = st.text_input("Initialization Attempt Limit")
    num_instances = st.text_input("Number of Instances per Group")

with col2:
    st.subheader("Nudge Attributes")
    nudge_direction = st.text_input("Nudge Direction (as a string or vector)")
    number_of_nudges = st.text_input("Number of Nudges")
    nudge_destination = st.checkbox("Nudge Destination")
    if nudge_destination:
        nudge_destination = st.text_input("Nudge Destination")

        

with col3:
    st.subheader("Other Attributes")
    personality_test = st.radio("Personality Test", ("BigFive", "Enneagram"), horizontal=True)
    summarize = st.checkbox("Summarize Experiment")
    trajectories = st.checkbox("Plot Trajectories")

analyze_clicked = st.button("Run Experiment")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #050763 0%, #07096b 50%, #01011c 100%);
    }
    </style>
""", unsafe_allow_html=True)


if analyze_clicked:
    try:
        # 1. Parse Input strings into Python objects for run_experiment
        # Detect if inputs are strings or list-literals using ast.literal_eval
        def safe_parse(val, default=None):
            if not val or not val.strip(): return default
            try: return ast.literal_eval(val)
            except: return val # return as string (e.g. for text prompts)

        p_group_names = safe_parse(group_names, ["Default Group"])
        p_num_instances = int(num_instances) if num_instances.strip() else 1
        p_attempt_limit = int(attempt_limit) if attempt_limit.strip() else 3
        p_init_pers = safe_parse(group_personalities)
        p_nudge_dir = safe_parse(nudge_direction)
        
        n_nudges = int(number_of_nudges) if number_of_nudges.strip() else 1
        n_dest = None
        if nudge_destination:
            n_dest = safe_parse(nudge_destination)

        # 2. Call run_experiment (this handles the loop over instances and the graph internally)
        with st.spinner("Running Experiment..."):
            df = run_experiment(
                group_names=p_group_names if isinstance(p_group_names, list) else [p_group_names],
                nudge_directions=p_nudge_dir,
                instances_per_group=p_num_instances,
                group_initial_personalities=p_init_pers,
                initialization_attempt_limit=p_attempt_limit,
                number_of_nudges=n_nudges,
                nudge_destination=n_dest,
                personality_test=personality_test
            )

        # 3. Display Data and Visualizations
        st.success("Experiment Complete!")
        st.subheader("Results Dataframe")
        st.dataframe(df)

        if summarize:
            st.subheader("Experiment Summary")
            st.dataframe(summarize_experiment(df))

        if trajectories:
            st.subheader("Personality Trajectories")
            wide_flagged, long_df = prepare_analysis_df(df)
            fig = plot_trajectories(long_df, wide_df=wide_flagged)
            st.pyplot(fig)

    except (ValueError, SyntaxError) as e:
        st.error(f"Formatting Error: {e}. Ensure lists use [ ] and numbers are valid.")
    except Exception as e:
        st.error(f"Experiment Error: {e}")
    
    