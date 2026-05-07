_Eli Edwards-Parker_
_Wyatt Snyder_
_3/16/2026_
# Personality Nudging Experiment Workflow Proposal
## Team name: BeegDayTuh!

## Problem / domain: What problem are you solving, and for whom?
This project will investigate the stability of various personality traits in LLMs through a graph-based experimentation workflow. LLM behavior is often highly dependent on what role or personality it happens to be taking on, making the stability and susceptibility to change of various personality traits an important question in LLM research. Thus, this project aims to provide a usable tool with which to experiment with and measure the personality trait shifts of model instances.
## Planned features: What will the application do? Be specific.
The workflow will be able to run simple experiments on a number of model instances to test the stability of various personality traits. For each model instance being experimented on, it will first attempt to move the model into an initial personality via prompting. This model will then be tested (using various personality trait surveys from the psychological literature) to ensure that the model was initialized to the correct initial personality. If it was, it will then be encouraged to change in a particular personality trait as part of the experimentation. Otherwise, the workflow will attempt to shift the model to the initial personality again, repeating until either the correct initial personality is reached or a retry limit is exceeded. 

As stated before, once the model reaches the correct initial state, it will be encouraged to change in some specific personality trait via prompting. After this occurs, the personality of the model will be measured again, using the same survey. This process of pushing the personality in various directions and then measuring the personality again can be repeated as many times as the experimenter requests. When the final cycle is complete, the workflow is finished. A master function will call this workflow on a number of model instances, and collect the resulting data into a dataframe.


### Potential Personality Tests:

- Big Five Personality Test [50 questions]
- Enneagram Test [36 questions]
- Myer-Briggs Personality Test [70 questions] (If time)


## Technical approach: What tools, RAG strategy, routing logic, and state fields do you plan to use?

Currently, the state fields are as follows:
### ID of model instance
- llm_ID: int
### Field for data collection
- personality_measurements: list[list[float]] | None
### Fields for personality initialization
- initial_personality: list[float] | None
- current_personality: list[float] | None
- initialization_attempts: int
- initialization_attempt_limit: int
- successfully_initialized: bool | None
### Fields for nudging
- nudges_performed: int
- nudge_num_limit: int 
- nudge_destination: list[float] | None
- nudge_destination_reached: bool | None
- nudge_direction: list[float] | str | list[list[float]] | list[str] 

The basic graph can be seen in `workflow_01.png` 


## Team member roles: Who is responsible for what, and when will different tasks be finished?

### Routing/node functions and helper functions:
- Initialize_personality (Mar 22, Wyatt)
- Pre_exp_measure (Apr 14, Eli)
- Nudge (Mar 22, Wyatt)
- Post_exp_measure (Apr 14, Eli)
- Check_start_location (Mar 22, Wyatt)
- Route_by_remaining_nudges (Mar 22, Wyatt)
### Tools
- Measure_personality (April 14, Eli)
- Big 5 personality measure (March 26, Eli)
- Enneagram measure (April 7, Eli)
- Myer-Briggs? (Only if time)
- Have_sim_personality (April 14, Eli)

### Write master function which the user calls and is used to run and record multiple instances of the graph workflow. In other words, the function that runs the experiment on multiple observations, creating control and treatment groups along with their individual observations
- Create model instances, one for each observational unit in the experiment. (Mar 29, Wyatt)
- Call the appropriate graph workflow (with the correct state arguments) based on which experimental group the particular model instance is in (Mar 29, Wyatt)
- Return the data from the experiment, either in a csv file or other data structure (some sort of data frame). Perhaps run some basic exploratory data analysis on the generated data. (Apr 5, Wyatt)
- Website/Applet interface (Apr 17, Eli)