## Entry 1 - workflow scaffolding

**Date:** 03/04/2026

**Goal:** Set up basic repository folder and file set up, so future uploads have something to work from. 

**Tool used:** None

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** added files and folders similar to those used in the paper finder/analysis assignment.

**Testing:** none performed here, besides checking structure against the original assignment.

**Result:** Basic workflow structure to build from an modify in the future.


## Entry 2 - 

**Date:** 03/16/2026

**Goal:** Add overall proposal for project to repository

**Tool used:** none

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** uploaded proposal.md into workflow

**Testing:** rendered.

**Result:** structure was functional.


## Entry 3 - first node implementations

**Date:** 03/31/2026

**Goal:** Start building first outlines of nodes needed for graph workflow

**Tool used:** Claude

**Prompt/Question:** went over basic suggestions for nodes needed as well as state fields that worked as effective complements of this. 

**AI Response:** approved of basic structure. suggestions were nonsyntactical

**Changes Made:** nodes.py, graph.py, experiment.py (more syntax suggestions were provided in this file than in some of the tothers), state.py

**Testing:** none yet, as nothing is functional enough to call yet

**Result:** bare bones implementations of some nodes created.


## Entry 4 - initialize personality

**Date:** 03/31/2026

**Goal:** implement nodes, continued
**Tool used:** none

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** experiment.py, deleting some old nodes and implementing initialize personality 

**Testing:** none at this point

**Result:** cleaned (somewhat) nodes.py, experiment.py, and implemented 'initialize personality'


## Entry 5 - comments

**Date:** 04/02/2026

**Goal:** add comments to all major files, describing high-level what is going on in code.

**Tool used:** none

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** nodes.py, state.py, tools.py, llm.py, graph.py, experiment.py

**Testing:** n/a

**Result:** clear comments on functions of major files, as well as some smaller-scale functions. 


## Entry 6 - incorporating memory into nodes

**Date:** 04/12/2026

**Goal:** make it so that calls to the llm are made utilizing previous conversation as context

**Tool used:** Claude

**Prompt/Question:** Here's the current implementation and state (...), how might I implement memory use into these llm calls? 

**AI Response:** Provided syntax allowing proper use of previous conversation responses and prompts stored in the state. Some updated the memory, but others didn't (i.e., measuring the llm personality prompts).

**Changes Made:** nodes.py, state.py

**Testing:** Verifying syntax with llm.

**Result:** with modifications, settled on accepted syntax.


## Entry 7 - Further editing of comments, deleted unneeded files

**Date:** 04/14/2026

**Goal:** Clean workflow, deleting unneeded files and adding comments

**Tool used:** none

**Prompt/Question:** n/a

**AI Response:** n/a.

**Changes Made:** graph.y, nodes.py, state.py, output.py (deleted), cli.py (deleted)

**Testing:** ensured no functions in files were being used

**Result:** deleted 2 files, with future todo of adding other user-interface files to interact with experimental workflow.


## Entry 8 - EDA functions for results

**Date:** 04/14/2026

**Goal:** Build an academic coach agent with access to a range of textbooks and some relavent mathematical functions

**Tool used:** Gemini Code Assist, Claude

**Prompt/Question:** explained data strcuture being provided from each state instance, and requested implementation to expand dataframe with vectors, as well as to visualize resulting data

**AI Response:** code alteration suggestions provided, within the repository framework. Used Gemini code assist within ide untill it stopped working. Then switched to Claude (bulk of work done here).

**Changes Made:** expiriment.py, added function_prereq_graph.svg (image showing which functions need results from which other functions).

**Testing:** checking via llm that calls were correct syntactically (as I wrote portions of them, particularly the portions involving parsing argument into different types).

**Result:** several functions performing the initial experiment, the data expansion, and some basic EDA, particularly visualization.


## Entry 9 - new node: conversational noise 

**Date:** 04/20/2026

**Goal:** Build a node that inserts normal conversation(?) into an experimental workflow using an llm.

**Tool used:** n/a

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** nodes.py. Added implementation of a new node, without working it into the workflow yet. Currently not functional, so more of a syntactically dubious implementation of what this might look like once functional. 

**Testing:** none- just rough draft

**Result:** rough implementation of node requesting from an llm a response/question for the llm being experimented on, based only on previous conversation.


## Entry 10 - finished personnality function and started interface 

**Date:** 03/14

**Goal:** implement functions measuring personality, and also start app for user interface

**Tool used:** n/a

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** tools.py. Added personality similarity function as well as personality measurement function. app.py/ Pasted document in from a9 to get started. requirements.txt. Added pydantic, numpy, and streamlit.

**Testing:** not testable yet - need workflow running

**Result:** n/a


## Entry 11 - new node: conversational noise (tool) 

**Date:** 04/21/2026

**Goal:** Build a node that inserts normal conversation(?) into an experimental workflow using an llm.

**Tool used:** N/A

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** nodes.py, tools.py. Continued implementation of a new node, without working it into the workflow yet. Currently not functional, so more of a syntactically dubious implementation of what this might look like once functional. Main new dev here was tools the node would use: generate noise prompt

**Testing:** none- just rough draft

**Result:** rough implementation of node requesting from an llm a response/question for the llm being experimented on, based only on previous conversation. tool generating this prompt and using it to invoke the llm.


## Entry 12 - comment update

**Date:** 04/21/2026

**Goal:** add comment detailing the graph structure at tope of nodes.py

**Tool used:** n/a

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** nodes.py. Added description of overarching graph structure. 

**Testing:** none

**Result:** extended comment detailing what happens in the main graph running the instance workflow


## Entry 13 - update state formation, import statements and initialization to reflect most recent workflow
**Date:** 04/21/2026

**Goal:** update state formation, import statements and initialization to reflect most recent workflow

**Tool used:** n/a

**Prompt/Question:** n/a

**AI Response:** n/a

**Changes Made:** nodes.py. Implemented personality function. tools.py. Added message history input to measure personality, but kept the actual test out of the history. app.py added to import statements, and fixed old state initialization calls to reflect new variables.

**Testing:** entire workflow needed to test

**Result:** n/a


## Entry 14 - new file to run experiment 
https://github.com/ai-in-the-loop-2026/final-project-working-beegdaytuh/commit/60db87b3ef130e707af36055aa4d93528ed59c4
**Date:** 04/26/2026

**Goal:** simply run and test entire workflow using separate python script.
**Tool used:** Claude

**Prompt/Question:** What's the simplest way to run this for testing purposes? 

**AI Response:** Gave three options, one of which was a script, for which it provided basic syntax

**Changes Made:** added testRun.py and provided basic implementation.

**Testing:** Feedback from model on implementation modifications

**Result:** implementation that should in theory work when run 


## Entry 15 - fix random data structure errors 
**Date:** 04/26/2026

**Goal:** fix data issues in workflow   

**Tool used:** ChatGpt, Claude

**Prompt/Question:** see any errors here (and similar prmopts)

**AI Response:** provided suggested edits, particularly in syntax around state fields

**Changes Made:** experiment.py, nodes.py, app.py

**Testing:** based on feedback from llms

**Result:** updated calls with more correct syntax


## Entry 16 - find and fix errors in files 
**Date:** 04/26/2026

**Goal:** find and fix as many errors as feasible

**Tool used:** Claude (mostly)- Gemini code assist wasn't working

**Prompt/Question:** point out errors across files and within implementations, suggesting fixes as needed. (context files provided)

**AI Response:** Many edit suggestions provided- some on critical implementation failures, others on improving robustness of implementation/readibility/professionalism/etc.

**Changes Made:** most major files underwent a number of changes

**Testing:** Feedback from models, often iterated several times

**Result:** much higher degree of cohesion between files and syntactically in general. Realizing now that spending a large chunk of time early on is often worth the effort and delay in initializing implementation, if only for the reason of preventing issues arising from not carefully thinking stuff through at the beginning. 


## Entry 17 - fix personality tool; implement streamlit app 
https://github.com/ai-in-the-loop-2026/final-project-working-beegdaytuh/commit/0b378f955bdda3916927c787fd3da31c0148bf2e
**Date:** 04/28/2026

**Goal:** develop a streamlit interface with which individuals can run the experiment and modify details

**Tool used:** claude

**Prompt/Question:** Asked claude about good formatting options for the streamlit interface.

**AI Response:** provided syntax for basic streamlit interface

**Changes Made:**  tools.py. Fixed 'reformer' typo. app.py. Created the 'shell' of the applet without any functionality

**Testing:** messed around with locations of each text box, and found a balance I liked (that later got changed :/)

**Result:** implementation of streamlit app


## Entry 18 - debugging more syntax issues 
**Date:** 04/28/2026

**Goal:** get the stinking thing to work

**Tool used:** claude, chatgpt

**Prompt/Question:** model inside vscode; worked back and forth to develop streamlit generating script (also worked with old versions from past assignments)

**AI Response:** found more issues, suggested implementations

**Changes Made:**  little things to many files regarding import statements (addding load_dotenv was helpful), also extra changes to experiment.py syntactically. There were a number of simple mistakes I repeatedly made regarding datastructure conformity that could have been avoided if I were a little more careful early on- something to keep in mind.

**Testing:** Tried to run again

**Result:** then I was just hitting max limit of calls exceeded errors, which suggested solid implementation in general


## Entry 19 - remove debugging stuff 
**Date:** 04/28/2026

**Goal:** Now that it worked, delete debugging print statements

**Tool used:** none

**Prompt/Question:** none

**AI Response:** none

**Changes Made:**  tools.py, state.py, testRun.py

**Testing:** ran experimental workflow, checked for unneed print statements

**Result:** clean and working repository!


## Entry 20 - fix app 
https://github.com/ai-in-the-loop-2026/final-project-working-beegdaytuh/commit/26df92057b5d46e328e395f3b79a387a83cbbdd8
**Date:** 04/29/2026

**Goal:** develop a streamlit interface with which individuals can run the experiment and modify details

**Tool used:** gemini code assist

**Prompt/Question:** questions about syntax and code organization. App was having issues with inputs

**AI Response:** Suggested a parse_input function that resolved all input issues and added default responses to many entries when left blank.

**Changes Made:**  new folder including folder with graphs produced; app.py. finalized the functionality of the app, testRun.py, altered implementation of streamlit app

**Testing:** clicked run experiment until the run experiment did what I wanted it to

**Result:** implementation of streamlit app that works!


