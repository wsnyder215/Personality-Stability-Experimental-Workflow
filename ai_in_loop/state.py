"""State schema for the personality experiment model instance workflow.

This module defines the ModelInstanceState, a TypedDict that flows through the LangGraph
workflow. Each node reads from and writes to fields in this state.
"""

from typing import TypedDict


class ModelInstanceState(TypedDict):
    """State object that flows through the experiment workflow for a particular model instance.

    The workflow attempts to nudge model instances with different initial personalities in different directions, and
    measures subsequent changes in personality. The state fields are designed to support this workflow, but also to 
    be general enough to support a variety of experiments with different personality measures, nudging strategies, 
    and LLMs.

    Fields are organized into logical groups:
    - ID and memory: info unique to the model instance and its interaction history
    - Experimental recording: variables for tracking measurements and responses during the experiment
    - initialization: variables related to initializing the instance's personality
    - nudge variables: variables related to carrying out nudges on model instance
    """
    
    #===model instance identification and memory===#
    llm_ID: int
    message_history: list[dict]

    #===experimental recording variables===#
    personality_test: str
    measurements: list[list[float]] | None
    responses: list[str]
    
    #===initialization variables===#
    initial_personality: list[float] | None
    current_personality: list[float] | None
    initialization_attempts: int
    initialization_attempt_limit: int
    successfully_initialized: bool | None
    
    #===nudge variables===#
    nudges_performed: int
    nudge_num_limit: int # note that this functions as the nudge limit for both instances
    # with and without a nudge_destination. Not sure if those should be separate fields.
    nudge_destination: list[float] | None
    nudge_destination_reached: bool | None
    nudge_direction: list[float] | str | list[list[float]] | list[str]
    warnings: list
    
    
    

