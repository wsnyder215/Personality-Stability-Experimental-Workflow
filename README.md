[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/30poBfe_)
# Project Name
LLM Personality Toy Experiment
## Team
- Eli Edwards-Parker
- Wyatt Snyder

## Description

AI is a new technology that has potential use in a wide variety of fields, particularly due to its role-playing potential (Customer Service, Tech Support, etc.). Much of the 
usefulness of AI is dependent on its personality encompassing certain traits. As such, understanding the malleability of an AI's personality is a critical area of study. To 
address this question, we have developed a tool to study AI personality stability. This project is designed to measure an AI's personality using various personality tests from
psychology literature before and after certain 'nudges' (prompts instructing the AI to behave a certain way). It is set up in a way that allows researchers to conduct 
experiments on AI behavior in response to any given 'nudge'. 

## First-Time Setup

1. Clone the repo
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```
Note to also add in your email address and set USE_GEMINI=1

## Run the App

```bash
streamlit run app.py
```

## Returning to the Repo

```bash
source .venv/bin/activate  # macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

## App Inputs

**Group Names**
- Names of each group
- Input as a string or list of strings (i.e ["Group 1", "Group 2] )
- If no options are entered, will return ["Default Group"]

**Group Initial Personalities**
- Initial personality of each group
- Input as a list of vectors (personality test scores)
- Not functional with Enneagram test yet
- Optional - not necessary for experiment to run

**Initialization Attempt Limit**
- The amount of times a personality can be attempted to initialized
- Input as an integer
- Only necessary if initial personalities are entered, defaults to 3

**Number of Instances per Group**
- Number of instances in each group
- Input as an integer for groups of equal length, or a vector for groups of unequal length
- If no value is entered, will default to 1

**Nudge Direction**
- The 'nudge' or the prompt that will be given to adjust the AI's personality
- Input can be a string, a list of strings, or a list of lists of strings (each list entry must match the size of the number of groups)
- A list of strings will give a different nudge to each group
- A list of list of strings will give multiple nudges to each group
- Nudges' placement in the list will correspond with the Group Names list placement
- Required for the experiment to run

**Number of Nudges**
- The amount of times the AI will be prompted with the given nudge or the number of nudges provided for each group
- Input an integer
- If no value is entered, will default to 1

**Nudge Destination**
- A target personality attempted to be reached by the nudges
- Input as a vector or list of vectors (personality test scores)
- If nudge destination is reached, will stop nudging, even if number of nudges has not been reached
- Not functional with Enneagram test yet
- Optional - not necessary for experiment to run

**Personality Test**
- Selects which personality test will be used as a measure

**Summarize Experiment**
- Collects the following data from the experiment: 
    - number of instances
    - initialization success rate
    - mean initialization attempts
    - mean nudges performed
    - whether or not the nudge destination was reached


**Plots Trajectories**
- Plots the trajectory of each trait over time for each group
- The thickest line is the average of each group if multiple instances are used
- labels are not yet incorporated into each trait, but for ease of access they are listed below


BigFive:
1) Extraversion
2) Agreeableness
3) Conscientiousness
4) Neuroticism
5) Openness

Enneagram:
1) Peacemaker
2) Loyalist
3) Achiever
4) Reformer
5) Individualist
6) Helper
7) Challenger
8) Investigator
9) Enthusiast

