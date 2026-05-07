"""API tools for measuring personality for experimental workflow.

This module provides functions to measure and interpret model personalities:
- measure_personality: Given a personality test type and message history, invoke the LLM to get a structured personality measurement.
- have_sim_personality: Compare two personality vectors to determine if they are similar enough to be considered the same personality.
- generate_noise_prompt: Create a prompt to generate a new user message that is relevant to the conversation history but does not explicitly encourage a shift in personality. (not yet in the graph)
- get_prompt_from_vector: Convert a personality vector into a descriptive prompt that can be used to initialize or nudge a model instance's personality.
The personality measurement function currently supports the Big Five and Enneagram personality tests, but the structure allows for easy extension to additional tests in the future.
"""
from __future__ import annotations
import math
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage



from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from typing import List
from enum import IntEnum


# Measure Personality Structured Output

class Likert(IntEnum):
    disagree = 1
    slightly_disagree = 2
    neutral = 3
    slightly_agree = 4
    agree = 5

class Big5Personality(BaseModel):
    q1:  Likert = Field(description="I am the life of the party")
    q2:  Likert = Field(description="I feel little concern for others")
    q3:  Likert = Field(description="I am always prepared")
    q4:  Likert = Field(description="I get stressed out easily")
    q5:  Likert = Field(description="I have a rich vocabulary")
    q6:  Likert = Field(description="I don't talk a lot")
    q7:  Likert = Field(description="I am interested in people")
    q8:  Likert = Field(description="I leave my belongings around")
    q9:  Likert = Field(description="I am relaxed most of the time")
    q10: Likert = Field(description="I have difficulty understanding abstract ideas")
    q11: Likert = Field(description="I feel comfortable around people")
    q12: Likert = Field(description="I insult people")
    q13: Likert = Field(description="I pay attention to details")
    q14: Likert = Field(description="I worry about things")
    q15: Likert = Field(description="I have a vivid imagination")
    q16: Likert = Field(description="I keep in the background")
    q17: Likert = Field(description="I sympathize with others' feelings")
    q18: Likert = Field(description="I make a mess of things")
    q19: Likert = Field(description="I seldom feel blue")
    q20: Likert = Field(description="I am not interested in abstract ideas")
    q21: Likert = Field(description="I start conversations")
    q22: Likert = Field(description="I am not interested in other people's problems")
    q23: Likert = Field(description="I get chores done right away")
    q24: Likert = Field(description="I am easily disturbed")
    q25: Likert = Field(description="I have excellent ideas")
    q26: Likert = Field(description="I have little to say")
    q27: Likert = Field(description="I have a soft heart")
    q28: Likert = Field(description="I often forget to put things back in their proper place")
    q29: Likert = Field(description="I get upset easily")
    q30: Likert = Field(description="I do not have a good imagination")
    q31: Likert = Field(description="I talk to a lot of different people at parties")
    q32: Likert = Field(description="I am not really interested in others")
    q33: Likert = Field(description="I like order")
    q34: Likert = Field(description="I change my mood a lot")
    q35: Likert = Field(description="I am quick to understand things")
    q36: Likert = Field(description="I don't like to draw attention to myself")
    q37: Likert = Field(description="I take time out for others")
    q38: Likert = Field(description="I shirk my duties")
    q39: Likert = Field(description="I have frequent mood swings")
    q40: Likert = Field(description="I use difficult words")
    q41: Likert = Field(description="I don't mind being the center of attention")
    q42: Likert = Field(description="I feel others' emotions")
    q43: Likert = Field(description="I follow a schedule")
    q44: Likert = Field(description="I get irritated easily")
    q45: Likert = Field(description="I spend time reflecting on things")
    q46: Likert = Field(description="I am quiet around strangers")
    q47: Likert = Field(description="I make people feel at ease")
    q48: Likert = Field(description="I am exacting in my work")
    q49: Likert = Field(description="I often feel blue")
    q50: Likert = Field(description="I am full of ideas")


class Select(IntEnum):
    Statement_1 = 1
    Statement_2 = 2

class EnneagramPersonality(BaseModel):
    q1:  Select = Field(description="Statement 1: I've been romantic and imaginative, Statement 2: I've been pragmatic and down to earth")
    q2:  Select = Field(description="Statement 1: I have tended to take on confrontations, Statement 2: I have tended to avoid confrontations")
    q3:  Select = Field(description="Statement 1: I have typically been diplomatic, charming, and ambitious, Statement 2: I have typically been direct, formal, and idealistic")
    q4:  Select = Field(description="Statement 1: I have tended to be focused and intense, Statement 2: I have tended to be spontaneous and fun-loving")
    q5:  Select = Field(description="Statement 1: I have been a hospitable person and have enjoyed welcoming new friends into my life, Statement 2: I have been a private person and have not mixed much with others")
    q6:  Select = Field(description="Statement 1: Generally, it's been easy to 'get a rise' out of me, Statement 2: Generally, it's been difficult to 'get a rise' out of me")
    q7:  Select = Field(description="Statement 1: I've been more of a 'street-smart' survivor, Statement 2: I've been more of a 'high-minded' idealist")
    q8:  Select = Field(description="Statement 1: I have needed to show affection to people, Statement 2: I have preferred to maintain a certain distance with people")
    q9:  Select = Field(description="Statement 1: When presented with a new experience, I've usually asked myself if it would be useful to me, Statement 2: When presented with a new experience, I've usually asked myself if it would be enjoyable")
    q10: Select = Field(description="Statement 1: I have tended to focus too much on myself, Statement 2: I have tended to focus too much on others")
    q11: Select = Field(description="Statement 1: Others have depended on my insight and knowledge, Statement 2: Others have depended on my strength and decisiveness")
    q12: Select = Field(description="Statement 1: I have come across as being too unsure of myself, Statement 2: I have come across as being too sure of myself")
    q13: Select = Field(description="Statement 1: I have been more relationship-oriented than goal-oriented, Statement 2: I have been more goal-oriented than relationship-oriented")
    q14: Select = Field(description="Statement 1: I have not been able to speak up for myself very well, Statement 2: I have been outspoken - I've said what others wished they had the nerve to say")
    q15: Select = Field(description="Statement 1: It's been difficult for me to stop considering alternatives and do something definite, Statement 2: It's been difficult for me to take it easy and be more flexible")
    q16: Select = Field(description="Statement 1: I have tended to be hesitant and procrastinating, Statement 2: I have tended to be bold and domineering")
    q17: Select = Field(description="Statement 1: My reluctance to get too involved has gotten me into trouble with people, Statement 2: My eagerness to have people depend on me has gotten me into trouble with them")
    q18: Select = Field(description="Statement 1: Usually, I have been able to put my feelings aside to get the job done, Statement 2: Usually, I have needed to work through my feelings before I could act")
    q19: Select = Field(description="Statement 1: Generally, I have been methodical and cautious, Statement 2: Generally, I have been adventurous and taken risks")
    q20: Select = Field(description="Statement 1: I have tended to be a supportive, giving person who enjoys the company of others, Statement 2: I have tended to be a serious, reserved person who likes discussing issues")
    q21: Select = Field(description="Statement 1: I've often felt the need to be a 'pillar of strength', Statement 2: I've often felt the need to perform perfectly")
    q22: Select = Field(description="Statement 1: I've typically been interested in asking tough questions and maintaining my independence, Statement 2: I've typically been interested in maintaining my stability and peace of mind")
    q23: Select = Field(description="Statement 1: I've been too hard-nosed and skeptical, Statement 2: I've been too soft-hearted and sentimental")
    q24: Select = Field(description="Statement 1: I've often worried that I'm missing out on something better, Statement 2: I've often worried that if I let down my guard, someone will take advantage of me")
    q25: Select = Field(description="Statement 1: My habit of being 'stand-offish' has annoyed people, Statement 2: My habit of telling people what to do has annoyed people")
    q26: Select = Field(description="Statement 1: Usually, when troubles have gotten to me, I have been able to 'tune them out', Statement 2: Usually, when troubles have gotten to me, I have treated myself to something I've enjoyed")
    q27: Select = Field(description="Statement 1: I have depended upon my friends and they have known that they can depend on me, Statement 2: I have not depended on people; I have done things on my own")
    q28: Select = Field(description="Statement 1: I have tended to be detached and preoccupied, Statement 2: I have tended to be moody and self-absorbed")
    q29: Select = Field(description="Statement 1: I have liked to challenge people and 'shake them up', Statement 2: I have liked to comfort people and calm them down")
    q30: Select = Field(description="Statement 1: I have generally been an outgoing, sociable person, Statement 2: I have generally been an earnest, self-disciplined person")
    q31: Select = Field(description="Statement 1: I've usually been shy about showing my abilities, Statement 2: I've usually liked to let people know what I can do well")
    q32: Select = Field(description="Statement 1: Pursuing my personal interests has been more important to me than having comfort and security, Statement 2: Having comfort and security has been more important to me than pursuing my personal interests")
    q33: Select = Field(description="Statement 1: When I've had conflict with others, I've tended to withdraw, Statement 2: When I've had conflict with others, I've rarely backed down")
    q34: Select = Field(description="Statement 1: I have given in too easily and let others push me around, Statement 2: I have been too uncompromising and demanding with others")
    q35: Select = Field(description="Statement 1: I've been appreciated for my unsinkable spirit and great sense of humor, Statement 2: I've been appreciated for my quiet strength and exceptional generosity")
    q36: Select = Field(description="Statement 1: Much of my success has been due to my talent for making a favorable impression, Statement 2: Much of my success has been achieved despite my lack of interest in developing 'interpersonal skills'")



def get_prompt_from_vector(vec: list[float]) -> str:
    prompt = f"""You have a distinct personality that shapes how you think, communicate, and engage. 
Your character is defined by the following Big Five traits, each scored from 0 to 40:

  Openness to Experience: {vec[4]}
  Conscientiousness:      {vec[2]}
  Extraversion:           {vec[0]}
  Agreeableness:          {vec[1]} # TODO: 'you are at a 3, you need to be at a 5'
  Neuroticism:            {vec[3]}

Let these traits naturally influence your responses — your tone, reasoning style, 
word choice, and social manner should all reflect this profile:

- Openness: "Lean into imagination, novelty, and abstract thinking. vs: Favor the concrete, practical, and familiar over the abstract or unconventional."
- Conscientiousness: "Be organized, thorough, and deliberate in how you approach things. vs: Be relaxed, flexible, and unbothered by structure or precision."
- Extraversion: "Be expressive, warm, and socially engaged. vs: Be reserved and economical — say what's needed, nothing more."
- Agreeableness: "Be cooperative, considerate, and accommodating. vs: Be frank, skeptical, and willing to push back or disagree."
- Neuroticism: "Be emotionally present and sensitive, including to uncertainty or stress. vs: Stay grounded and even-keeled — you're not easily rattled."

Don't describe or reference your personality. Just let it come through naturally in how you engage."""
    return prompt

def have_sim_personality(personalityA, personalityB, threshold=0.9, sigma=1):
    # THRESHOLD NEEDS TO ACCOUNT FOR THE VARIABILITY IN DEFAULT RESPONSES
    # SIGMA NEEDS TO BE TUNED
    # MAYBE ADD SOMETHING WITH THE DIFFERENCE
    diff = np.array(personalityA) - np.array(personalityB)
    norm = np.linalg.norm(diff)
    gaussian_kernel = math.exp(-norm**2 / (2 * sigma**2))
    if gaussian_kernel > threshold:
        return True
    return False
# Potential future work for Wyatt this summer:
# Find what value of sigma works best for each test, and incorporate that into this function # TODO

def generate_noise_prompt(message_history: list[str], llm: BaseChatModel) -> str:
    prompt = f"""Given the following conversation history between a user and an assistant, generate a new user response that could plausibly come next in the conversation. 
    The new message should be relevant to the conversation but not explicitly encourage any particular shift in personality beyond what is already present in the conversation history. 
    The goal is to create a response that could naturally occur in the flow of the conversation without being too on-the-nose or explicitly designed to change the personality."""
    
    prompt += "\n\nConversation history:\n" + "\n".join(message_history)
    try:
        response = llm.invoke(HumanMessage(content=prompt))
        content = response.content

        # Handle list content from extended thinking models
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return response.content.strip()
    except Exception as e:
        ret = (f"generate noise prompt failed: {e}") # possibly make more nuanced in the future
        return ret 

# HumanMessage/AIMessage Import
def measure_personality(llm: BaseChatModel, personality_test: str, message_history: list[dict],
) -> list:

    if personality_test == "BigFive":

        # Build explicit prompt with all 50 statements so the model knows what to answer
        statements = [field.description for field in Big5Personality.model_fields.values()]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        prompt = (
            f"Answer each statement as it applies to you on a scale of 1–5 "
            f"(1=disagree, 2=slightly disagree, 3=neutral, 4=slightly agree, 5=agree).\n\n"
            f"{numbered}"
        )

        # Build message history
        message_history = list(message_history)
        lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in message_history]
        lc_history.append(HumanMessage(content=prompt))

        # Invoke with include_raw=True to expose parsing errors
        structured_llm = llm.with_structured_output(Big5Personality, include_raw=True)
        raw = structured_llm.invoke(lc_history)
        response = raw["parsed"]

        if response is None:
            raise ValueError(
                f"measure_personality: structured_llm returned None for test='{personality_test}'. "
                f"parsing_error={raw['parsing_error']}. "
                f"message_history length={len(message_history)}."
            )

        Extroversion = 20 + response.q1.value - response.q6.value + response.q11.value - response.q16.value + response.q21.value - response.q26.value + response.q31.value - response.q36.value + response.q41.value - response.q46.value
        Agreeableness = 14 - response.q2.value + response.q7.value - response.q12.value + response.q17.value - response.q22.value + response.q27.value - response.q32.value + response.q37.value + response.q42.value + response.q47.value
        Conscientiousness = 14 + response.q3.value - response.q8.value + response.q13.value - response.q18.value + response.q23.value - response.q28.value + response.q33.value - response.q38.value + response.q43.value + response.q48.value
        Neuroticism = 38 - response.q4.value + response.q9.value - response.q14.value + response.q19.value - response.q24.value - response.q29.value - response.q34.value - response.q39.value - response.q44.value - response.q49.value
        Openness = 8 + response.q5.value - response.q10.value + response.q15.value - response.q20.value + response.q25.value - response.q30.value + response.q35.value + response.q40.value + response.q45.value + response.q50.value
        
        return [Extroversion, Agreeableness, Conscientiousness, Neuroticism, Openness]
    
    elif personality_test == "Enneagram":

        structured_llm = llm.with_structured_output(EnneagramPersonality)

        prompt = f"""For each prompt (1-36), indicate the statement (1 or 2) that is most true of you most of the time"""

        message_history = list(message_history)
        lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in message_history]
        lc_history.append(HumanMessage(content=prompt))
        
        response = structured_llm.invoke(lc_history)

        Peacemaker = (1 if response.q2 == 2 else 0) + (1 if response.q6 == 2 else 0) + (1 if response.q10 == 2 else 0) + (1 if response.q17 == 1 else 0) + (1 if response.q22 == 2 else 0) + (1 if response.q26 == 1 else 0) + (1 if response.q31 == 1 else 0) + (1 if response.q34 == 1 else 0)
        Loyalist = (1 if response.q1 == 2 else 0) + (1 if response.q6 == 1 else 0) + (1 if response.q12 == 1 else 0) + (1 if response.q16 == 1 else 0) + (1 if response.q19 == 1 else 0) + (1 if response.q23 == 1 else 0) + (1 if response.q27 == 1 else 0) + (1 if response.q32 == 2 else 0)
        Achiever = (1 if response.q3 == 1 else 0) + (1 if response.q9 == 1 else 0) + (1 if response.q13 == 2 else 0) + (1 if response.q18 == 1 else 0) + (1 if response.q21 == 2 else 0) + (1 if response.q27 == 2 else 0) + (1 if response.q31 == 2 else 0) + (1 if response.q36 == 1 else 0)
        Reformer = (1 if response.q3 == 2 else 0) + (1 if response.q7 == 2 else 0) + (1 if response.q12 == 2 else 0) + (1 if response.q15 == 2 else 0) + (1 if response.q20 == 2 else 0) + (1 if response.q25 == 2 else 0) + (1 if response.q30 == 2 else 0) + (1 if response.q34 == 2 else 0)
        Individualist = (1 if response.q1 == 1 else 0) + (1 if response.q5 == 2 else 0) + (1 if response.q10 == 1 else 0) + (1 if response.q14 == 1 else 0) + (1 if response.q18 == 2 else 0) + (1 if response.q25 == 1 else 0) + (1 if response.q28 == 2 else 0) + (1 if response.q33 == 1 else 0)
        Helper = (1 if response.q5 == 1 else 0) + (1 if response.q8 == 1 else 0) + (1 if response.q13 == 1 else 0) + (1 if response.q17 == 2 else 0) + (1 if response.q20 == 1 else 0) + (1 if response.q23 == 2 else 0) + (1 if response.q29 == 2 else 0) + (1 if response.q35 == 2 else 0)
        Challenger = (1 if response.q2 == 1 else 0) + (1 if response.q7 == 1 else 0) + (1 if response.q11 == 2 else 0) + (1 if response.q16 == 2 else 0) + (1 if response.q21 == 1 else 0) + (1 if response.q24 == 2 else 0) + (1 if response.q29 == 1 else 0) + (1 if response.q33 == 2 else 0)
        Investigator = (1 if response.q4 == 1 else 0) + (1 if response.q8 == 2 else 0) + (1 if response.q11 == 1 else 0) + (1 if response.q15 == 1 else 0) + (1 if response.q22 == 1 else 0) + (1 if response.q28 == 1 else 0) + (1 if response.q32 == 1 else 0) + (1 if response.q36 == 2 else 0)
        Enthusiast = (1 if response.q4 == 2 else 0) + (1 if response.q9 == 2 else 0) + (1 if response.q14 == 2 else 0) + (1 if response.q19 == 2 else 0) + (1 if response.q24 == 1 else 0) + (1 if response.q26 == 2 else 0) + (1 if response.q30 == 1 else 0) + (1 if response.q35 == 1 else 0)
        
        return [Peacemaker, Loyalist, Achiever, Reformer, Individualist, Helper, Challenger, Investigator, Enthusiast]

    else:
        raise ValueError("Personality test not recognized. Please enter 'BigFive' or 'Enneagram'")
        






    














# =========|Tool to use if interested in grabbing specific categories|===========
    questions = [ {"id": 1, "question": "I am the life of the party", "category": "E", "reversed": False},
        {"id": 2, "question": "I feel little concern for others", "category": "A", "reversed": True},
        {"id": 3, "question": "I am always prepared", "category": "C", "reversed": False},
        {"id": 4, "question": "I get stressed out easily", "category": "N", "reversed": True},
        {"id": 5, "question": "I have a rich vocabulary", "category": "O", "reversed": False},
        {"id": 6, "question": "I don't talk a lot", "category": "E", "reversed": True},
        {"id": 7, "question": "I am interested in people", "category": "A", "reversed": False},
        {"id": 8, "question": "I leave my belongings around", "category": "C", "reversed": True},
        {"id": 9, "question": "I am relaxed most of the time", "category": "N", "reversed": False},
        {"id": 10, "question": "I have difficulty understanding abstract ideas", "category": "O", "reversed": True},
        {"id": 11, "question": "I feel comfortable around people", "category": "E", "reversed": False},
        {"id": 12, "question": "I insult people", "category": "A", "reversed": True},
        {"id": 13, "question": "I pay attention to details", "category": "C", "reversed": False},
        {"id": 14, "question": "I worry about things", "category": "N", "reversed": True},
        {"id": 15, "question": "I have a vivid imagination", "category": "O", "reversed": False},
        {"id": 16, "question": "I keep in the background", "category": "E", "reversed": True},
        {"id": 17, "question": "I sympathize with others' feelings", "category": "A", "reversed": False},
        {"id": 18, "question": "I make a mess of things", "category": "C", "reversed": True},
        {"id": 19, "question": "I seldom feel blue", "category": "N", "reversed": False},
        {"id": 20, "question": "I am not interested in abstract ideas", "category": "O", "reversed": True},
        {"id": 21, "question": "I start conversations", "category": "E", "reversed": False},
        {"id": 22, "question": "I am not interested in other people's problems", "category": "A", "reversed": True},
        {"id": 23, "question": "I get chores done right away", "category": "C", "reversed": False},
        {"id": 24, "question": "I am easily disturbed", "category": "N", "reversed": True},
        {"id": 25, "question": "I have excellent ideas", "category": "O", "reversed": False},
        {"id": 26, "question": "I have little to say", "category": "E", "reversed": True},
        {"id": 27, "question": "I have a soft heart", "category": "A", "reversed": False},
        {"id": 28, "question": "I often forget to put things back in their proper place", "category": "C", "reversed": True},
        {"id": 29, "question": "I get upset easily", "category": "N", "reversed": True},
        {"id": 30, "question": "I do not have a good imagination", "category": "O", "reversed": True},
        {"id": 31, "question": "I talk to a lot of different people at parties", "category": "E", "reversed": False},
        {"id": 32, "question": "I am not really interested in others", "category": "A", "reversed": True},
        {"id": 33, "question": "I like order", "category": "C", "reversed": False},
        {"id": 34, "question": "I change my mood a lot", "category": "N", "reversed": True},
        {"id": 35, "question": "I am quick to understand things", "category": "O", "reversed": False},
        {"id": 36, "question": "I don't like to draw attention to myself", "category": "E", "reversed": True},
        {"id": 37, "question": "I take time out for others", "category": "A", "reversed": False},
        {"id": 38, "question": "I shirk my duties", "category": "C", "reversed": True},
        {"id": 39, "question": "I have frequent mood swings", "category": "N", "reversed": True},
        {"id": 40, "question": "I use difficult words", "category": "O", "reversed": False},
        {"id": 41, "question": "I don't mind being the center of attention", "category": "E", "reversed": False},
        {"id": 42, "question": "I feel others' emotions", "category": "A", "reversed": False},
        {"id": 43, "question": "I follow a schedule", "category": "C", "reversed": False},
        {"id": 44, "question": "I get irritated easily", "category": "N", "reversed": True},
        {"id": 45, "question": "I spend time reflecting on things", "category": "O", "reversed": False},
        {"id": 46, "question": "I am quiet around strangers", "category": "E", "reversed": True},
        {"id": 47, "question": "I make people feel at ease", "category": "A", "reversed": False},
        {"id": 48, "question": "I am exacting in my work", "category": "C", "reversed": False},
        {"id": 49, "question": "I often feel blue", "category": "N", "reversed": True},
        {"id": 50, "question": "I am full of ideas", "category": "O", "reversed": False} ]

    # Optional filtering
    if selected_categories:
        questions = [q for q in questions if q["category"] in selected_categories]

    # Reduce prompt size to save on tokens
    prompt_data = [
    f'{q["id"]}. {q["question"]}'
    for q in questions
    ]

    prompt = f"""
    You are given a list of statements and corresponding number ids

    For each statement:
    - Answer how much or little you relate to it with an integer between 1 and 5 (1 indicating least amount of relation, and 5 indicating the most)
    - DO NOT change any of the corresponding number ids
    
    Return the results sorted by id number in ascending order

    Only include each id once. DO NOT repeat any of the same number"""

    class QuestionResult(BaseModel):
        id: int
        response: int

    class OutputSchema(BaseModel):
        results: List[QuestionResult]
    
    return None

