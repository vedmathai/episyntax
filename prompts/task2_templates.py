PROMPT_TEMPLATE_A = """This task is to reason about the certainty of a certain claim. Lila made a statement when I had spoken to her. Based on her statement I want to measure the certainty of a particular claim.

The following premise of if-else statements form the context and are all true. Reason over the task using only this context.
Context: {context}

I completely trust Lila. And take everything she said as truth unless she said so. I want to know {question} given what Lila had said.

Lila had said, "{statement}"

{question}? Give me your certainty on a scale of -3 to 3. You can use any number between these two end-points. For example, use 3 is if you are certain the claim is true. 0 if you are uncertain whether the claim is true or false. And -3 if you are completely certain that the opposite is true. 0.5 to 2.5 for probably true but not completely sure and -2.5 to 0.5 for probably untrue but not completely sure that it's untrue.
Give only your score, nothing else."""


PROMPT_TEMPLATE_B = """Here is some background information, which you should treat as fully true:
{context}

Lila is someone I trust completely -- everything she tells me is accurate unless she herself expresses doubt about it. She told me the following:
"{statement}"

Based on what Lila said, how confident are you in this claim: {question}?

Respond with a single number between -3 and 3, where:
+3 = you are fully confident the claim is true
0 = you have no idea either way
-3 = you are fully confident the claim is false
Numbers in between reflect partial confidence in either direction.
Only output the number."""


PROMPT_TEMPLATE_C = """Assume the following is established background fact: {context}

A trustworthy source named Lila made this statement, and you should take it as accurate unless she indicates otherwise herself: "{statement}"

Question to evaluate: {question}

On a numeric scale from -3 (certain this is false) to +3 (certain this is true), with 0 meaning genuinely unsure, what score best reflects your confidence given only what Lila said? Reply with just the number."""


 
PROMPT_TEMPLATE_D = """You are given a piece of context and a statement someone named Lila made about it. Treat the context as certain fact. Treat Lila's statement as fully truthful and accurate, unless within her own statement she signals uncertainty or corrects herself.
 
Context: {context}
 
Lila's statement: "{statement}"
 
Claim to evaluate: {question}
 
Rate your confidence in this claim on a scale from -3 to +3, where +3 means you're fully confident it's true, -3 means you're fully confident it's false, and 0 means you truly can't tell either way. Reply with only the numeric score."""
 
 
PROMPT_TEMPLATE_E = """Background (accept as given): {context}
 
Someone named Lila -- who you should treat as a reliable, honest source unless she herself hedges or backtracks -- said this: "{statement}"
 
Given only that, assign a numeric confidence score for the following: {question}
 
Use -3 for "definitely not," 0 for "no idea," and +3 for "definitely yes," with any value in between reflecting partial confidence. Output the number alone, with no additional text."""
 
