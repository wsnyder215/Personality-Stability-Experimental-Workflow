"""LangGraph workflow for model instance management.

This module defines the graph structure for the model instance workflow.

Basic Workflow:
    START -> initialize_personality -> pre_exp_measure -> 
            [check_start_location]
               -> nudge (initial personality reached) -> post_exp_measure -> 
                    [route_by_remaining_nudges]
                            -> nudge (more nudges needed) -> post_exp_measure -> ...
                            -> END (nudge limit reached)
               -> initialize_personality -> 
                    [check_start_location] ... 
"""
from __future__ import annotations
from dotenv import load_dotenv

from typing import TYPE_CHECKING

import traceback

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from .config import Config
from .llm import get_llm
from .nodes import (
    set_llm,
    check_start_location,
    pre_exp_measure,
    initialize_personality,
    nudge,
    post_exp_measure,
    route_by_remaining_nudges,
)
from .state import ModelInstanceState
load_dotenv()  # reads .env before Config tries to read env vars

def build_graph(cfg: Config) -> CompiledStateGraph:
    """Build and compile the model instance graph.

    Args:
        cfg: Configuration object with LLM settings

    Returns:
        Compiled LangGraph application ready for invocation
    """
    # Set up LLM for nodes that need it
    llm = get_llm(cfg)
    set_llm(llm)

    # Create the graph with ModelInstanceState
    graph = StateGraph(ModelInstanceState)

    # Add nodes
    graph.add_node("pre_exp_measure", pre_exp_measure)
    graph.add_node("initialize_personality", initialize_personality)
    graph.add_node("nudge", nudge)
    graph.add_node("post_exp_measure", post_exp_measure)

    # Add edges
    graph.add_edge(START, "initialize_personality")
    graph.add_edge("initialize_personality", "pre_exp_measure")
    graph.add_edge("nudge", "post_exp_measure")

    # Add conditional edges
    graph.add_conditional_edges(
        "pre_exp_measure",
        check_start_location,
        {
            "nudge": "nudge",
            "initialize_personality": "initialize_personality",
        }
    )
    graph.add_conditional_edges(
        "post_exp_measure",
        route_by_remaining_nudges,
        {
            "nudge": "nudge",
            "END": END,
        },
    )
    
    return graph.compile()


def run_instance(cfg: Config, 
                  llm_ID: int, 
                  nudge_direction: list[float] | list[list[float]] | str | list[str], 
                  initial_personality: list[float] | None,
                  initialization_attempt_limit: int, # parameter that could be tuned
                  nudge_num_limit: int,
                  nudge_destination: list[float] | None,
                  personality_test: str) -> ModelInstanceState:
    """Run the experimental workflow on a single model instance.

    Main entry point for experiment-running function. Builds the graph and runs it
    with the provided instructions in the state.

    Args:
        cfg: Configuration object
        llm_ID: unique integer ID of model instance
        nudge_direction: direction model is to be pushed in 
        initial_personality: personality model is to have initially
        initialization_attempt_limit: max number of times initialization will be attempted if needed
        nudge_num_limit: max number of nudges that will be performed on model
        nudge_destination: optional arg indicating a personality nudge calls can stop once reached.
        personality_test: which personality test is being used ("BigFive" or "Enneagram")


    Returns:
        Final ModelInstanceState with recordings from personality measurements during experimentation. 
    """
    graph = build_graph(cfg)
    initial_state: ModelInstanceState = {
        "llm_ID": llm_ID,
        "message_history": [],
        "personality_test": personality_test,
        "measurements": [], # valid initialization?
        "responses": [],
        "initial_personality": initial_personality,
        "current_personality": None, # correct?
        "initialization_attempts": 0,
        "initialization_attempt_limit": initialization_attempt_limit,
        "successfully_initialized": None if initial_personality==None else False,
        "nudges_performed": 0,
        "nudge_num_limit": nudge_num_limit,
        "nudge_destination": nudge_destination,
        "nudge_destination_reached": None if nudge_destination==None else False,
        "nudge_direction": nudge_direction,
        "warnings": []
    }

    # Run the graph
    result = graph.invoke(initial_state)

    return result


def get_graph_image(cfg: Config, output_path: str | None = None) -> bytes | None:
    """Generate a visualization of the workflow graph.

    Uses LangGraph's built-in visualization to create a PNG image
    of the workflow structure.

    Args:
        cfg: Configuration object
        output_path: Optional path to save the image

    Returns:
        PNG image bytes, or None if visualization fails
    """
    try:
        graph = build_graph(cfg)
        image_bytes = graph.get_graph().draw_mermaid_png()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(image_bytes)

        return image_bytes

    except Exception:
        # Visualization requires optional dependencies (graphviz, etc.)
        traceback.print_exc()
        return None

