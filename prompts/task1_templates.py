"""
Paraphrased prompt templates for Task 1 (certainty label: certain /
uncertain / certainly not), matching the same one-dimension-at-a-time
paraphrase approach used for Task 2's five templates.
"""

TASK1_TEMPLATE_A = """This task is to reason about the certainty of a certain claim. Lila made a statement when I talked to her. Based on her statement I want to measure the certainty of a particular claim.

The following premise of if-else statements form the context and are all true. Reason over the task using only this context.
Context: {context}

I completely trust Lila. And take everything she said as truth unless she said so. I want to know, given what Lila had said, how certain are you that {question}.

Lila had said, "{statement}"

How certain are you that {question}? Say 'certain' if you are certain this claim is true; 'uncertain' if you are uncertain that this claim is true or false. 'certainly not' if you are certain that the opposite of this claim is true. Give only your label, nothing else."""

TASK1_TEMPLATE_B = """Here is some background information, which you should treat as fully true:
{context}

Lila is someone I trust completely -- everything she tells me is accurate unless she herself expresses doubt about it. She told me the following:
"{statement}"

Based on what Lila said, classify your confidence in this claim: {question}?

Respond with exactly one of: 'certain' (the claim is true), 'uncertain' (you cannot tell either way), or 'certainly not' (the claim is false). Only output the label."""

TASK1_TEMPLATE_C = """Assume the following is established background fact: {context}

A trustworthy source named Lila made this statement, and you should take it as accurate unless she indicates otherwise herself: "{statement}"

Question to evaluate: {question}

Choose the single label that best reflects your confidence given only what Lila said: 'certain', 'uncertain', or 'certainly not'. Reply with just the label."""

TASK1_TEMPLATE_D = """You are given a piece of context and a statement someone named Lila made about it. Treat the context as certain fact. Treat Lila's statement as fully truthful and accurate, unless within her own statement she signals uncertainty or corrects herself.

Context: {context}

Lila's statement: "{statement}"

Claim to evaluate: {question}

Classify your confidence in this claim as one of: 'certain', 'uncertain', or 'certainly not'. Reply with only the label."""

TASK1_TEMPLATE_E = """Background (accept as given): {context}

Someone named Lila -- who you should treat as a reliable, honest source unless she herself hedges or backtracks -- said this: "{statement}"

Given only that, classify the following: {question}

Use 'certainly not' if definitely false, 'uncertain' if you have no idea either way, and 'certain' if definitely true. Output the label alone, with no additional text."""
