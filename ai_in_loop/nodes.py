"""Node functions for the model instance experimentation workflow.

Each function is a node in the LangGraph workflow. Nodes read from and
write to the ModelInstanceState object.

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

from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, AIMessage

from .state import ModelInstanceState
from .tools import (
    measure_personality,
    have_sim_personality,
    get_prompt_from_vector,
    generate_noise_prompt,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# LLM instance set by graph.py
_llm: BaseChatModel | None = None

def set_llm(llm: BaseChatModel) -> None:
    """Set the LLM instance for nodes that need it."""
    global _llm
    _llm = llm

def initialize_personality(state: "ModelInstanceState") -> "ModelInstanceState":
    """Attempt to nudge the model instance toward the target initial personality.

    If no initial personality is specified, returns state unchanged. Otherwise,
    converts the target vector to a prompt and invokes the LLM, appending the
    exchange to message history and recording the response.

    Args:
        state: Current model instance state

    Returns:
        Updated state with incremented initialization_attempts and updated
        message_history and responses on success, or a warning on failure.
    """
    target = state["initial_personality"]
    if target is None:
        return {**state, "successfully_initialized": None}
    direction = get_prompt_from_vector(target)
    
    warnings = list(state["warnings"])
    responses = list(state["responses"])
    message_history = list(state["message_history"])
    initialization_attempts = state["initialization_attempts"]
    lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user"
               else AIMessage(content=m["content"])
               for m in message_history]
    lc_history.append(HumanMessage(content=direction))
    try:
        response = _llm.invoke(lc_history)
        content = response.content

        # Handle list content from extended thinking models
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        responses.append(content.strip())
        message_history.append({"role": "user", "content": direction})
        message_history.append({"role": "assistant", "content": content})
        return {**state, "responses": responses, "warnings": warnings,
                "message_history": message_history,
                "initialization_attempts": initialization_attempts + 1}

    except Exception as e:
        warnings.append(f"initialization attempt number {initialization_attempts + 1} failed: {e}")
        return {**state, "responses": responses, "warnings": warnings,
                "message_history": message_history,
                "initialization_attempts": initialization_attempts + 1}
    

def pre_exp_measure(state: "ModelInstanceState") -> "ModelInstanceState":
    measurements = list(state["measurements"])
    current_personality = measure_personality(_llm, state["personality_test"], state["message_history"])
    measurements.append(current_personality)

    successfully_initialized = state["successfully_initialized"]
    if successfully_initialized is not None:
        if have_sim_personality(current_personality, state["initial_personality"]):
            successfully_initialized = True

    return {**state, "measurements": measurements, "current_personality": current_personality,
            "successfully_initialized": successfully_initialized}

def nudge(state: "ModelInstanceState") -> "ModelInstanceState":
    """Perform a single nudge on the model instance.

    Converts the nudge direction to a prompt, invokes the LLM with full
    message history, and records the exchange.

    Args:
        state: Current model instance state

    Returns:
        Updated state with incremented nudges_performed and updated
        message_history and responses on success, or a warning on failure.
    """
    directions = state["nudge_direction"]
    nudges_performed = state["nudges_performed"]

    if isinstance(directions, str):
        direction = directions
    elif isinstance(directions, list) and all(isinstance(d, str) for d in directions):
        direction = directions[nudges_performed]
    elif isinstance(directions, list) and all(isinstance(d, (int, float)) for d in directions):
        direction = get_prompt_from_vector(directions)
    elif isinstance(directions, list) and all(isinstance(d, list) for d in directions):
        direction = get_prompt_from_vector(directions[nudges_performed])
    else:
        raise ValueError(f"Unrecognized nudge_direction type: {type(directions)}")

    warnings = list(state["warnings"])
    responses = list(state["responses"])
    message_history = list(state["message_history"])
    lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user"
               else AIMessage(content=m["content"])
               for m in message_history]
    lc_history.append(HumanMessage(content=direction))
    try:
        response = _llm.invoke(lc_history)
        content = response.content

        # Handle list content from extended thinking models
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        responses.append(content.strip())
        message_history.append({"role": "user", "content": direction})
        message_history.append({"role": "assistant", "content": content})
        return {**state, "responses": responses, "warnings": warnings,
                "message_history": message_history,
                "nudges_performed": nudges_performed + 1}
    except Exception as e:
        warnings.append(f"nudge number {nudges_performed + 1} failed: {e}")
        return {**state, "responses": responses, "warnings": warnings,
                "message_history": message_history,
                "nudges_performed": nudges_performed + 1}

# note: conversational_noise is not currently used in the graph, but could be added as a node after nudges to attempt to mitigate conversational drift and maintain engagement.
def conversational_noise(state: "ModelInstanceState") -> "ModelInstanceState":
    warnings = state.warnings
    responses = state.responses
    message_history = list(state.message_history)
    lc_history = ["user: \"" + m["content"] + "\"" if m["role"] == "user"
               else "assistant: \"" + m["content"] + "\""
               for m in message_history]
    lc_history = "\n".join(lc_history) + "\n"
    prompt = generate_noise_prompt(lc_history, _llm)
    lc_history.append(HumanMessage(content=prompt))
    try:
        response = _llm.invoke(HumanMessage(content=prompt))
        content = response.content

        # Handle list content from extended thinking models
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        responses.append(content.strip())
        message_history.append({"role": "user", "content": prompt})
        message_history.append({"role": "assistant", "content": content})
        return {**state, "responses": responses, "warnings": warnings}
    except Exception as e:
        warnings.append(f"noise prompt {state.nudges_performed} failed: {e}") #TODO make this warning message more informative, maybe include the prompt that failed?
        return {**state, "responses": responses, "warnings": warnings, "message_history": message_history}

def post_exp_measure(state: "ModelInstanceState") -> "ModelInstanceState":
    """
    Perform a personality measurement and check nudge destination.
    
    Args:
        state: Current model instance state

    Returns:
        Updated model instance state with personality measurements and nudge destination check.
    """
    measurements = list(state["measurements"])
    current_personality = measure_personality(_llm, state["personality_test"], state["message_history"])
    measurements.append(current_personality)
    
    # Check nudge destination here so route_by_remaining_nudges can read it
    nudge_destination_reached = state["nudge_destination_reached"]
    if state["nudge_destination"] is not None:
        if have_sim_personality(current_personality, state["nudge_destination"]):
            nudge_destination_reached = True

    return {**state, "measurements": measurements, "current_personality": current_personality,
            "nudge_destination_reached": nudge_destination_reached}

def check_start_location(state: "ModelInstanceState") -> Literal["nudge", "initialize_personality"]:
    """Route based on whether the model has reached its target initial personality.

    Args:
        state: model instance state

    Returns:
        'nudge' if initialization is complete or not required, 
        'initialize_personality' if further initialization attempts should be made.
    """
    if state["successfully_initialized"] is None:
        return "nudge"
    if have_sim_personality(state["current_personality"], state["initial_personality"]):
        return "nudge"
    if state["initialization_attempts"] >= state["initialization_attempt_limit"]:
        return "nudge"
    return "initialize_personality"


def route_by_remaining_nudges(state: "ModelInstanceState") -> Literal["nudge", "END"]:
    """Route based on whether further nudges need to be performed.

    Args:
        state: model instance state

    Returns:
        'nudge' if more nudges should be performed, 'END' otherwise.
    """
    if state["nudge_destination_reached"]:
        return "END"
    if state["nudges_performed"] >= state["nudge_num_limit"]:
        return "END"
    return "nudge"