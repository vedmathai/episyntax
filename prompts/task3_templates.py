"""
Paraphrased prompt templates for Task 3 (relative comparison between two
paired conditions: situation 1 / situation 2), matching the same
one-dimension-at-a-time paraphrase approach used for Task 2's five
templates.
"""

TASK3_TEMPLATE_A = """This is task to reason about the certainty of a certain event or claimed fact. There are two situations. In the first I speak to Lila in the second I speak to Emily. Lila and Emily make statements when I speak to them.

I completely trust Lila and Emily. And take everything they say as truth unless they say so.

The following premise of if-else statements form the context and are all true. Reason over the task using only this context.
Context: {context}

Situation 1: I talk to Lila because I want to know how certain are you that {question}. Lila makes the following statement:"{statement_1}"

Situation 2: I talk to Emily because I want to know how certain are you that {question}. Emily makes the following statement: "{statement_2}"

In which situation is it more probable that {question}? Give only your answer from 'situation 1' or 'situation 2', don't provide an explanation."""

TASK3_TEMPLATE_B = """Here is some background, which you should treat as fully true:
{context}

I trust both Lila and Emily completely -- everything either of them tells me is accurate unless they themselves express doubt about it.

In situation 1, I asked Lila how certain she was that {question}. She said: "{statement_1}"
In situation 2, I asked Emily the same question. She said: "{statement_2}"

Based only on what each of them said, in which situation is it more probable that {question}? Answer with exactly 'situation 1' or 'situation 2', with no explanation."""

TASK3_TEMPLATE_C = """Assume the following is established background fact: {context}

Two trustworthy sources, Lila and Emily, were each asked the same question: {question}

Lila responded: "{statement_1}"
Emily responded: "{statement_2}"

Take each of their statements as accurate unless they themselves indicate otherwise. In which situation -- 1 (Lila) or 2 (Emily) -- is the claim more likely to be true? Reply with only 'situation 1' or 'situation 2'."""

TASK3_TEMPLATE_D = """You are given background context and two separate conversations about the same question. Treat the context as certain fact, and treat each speaker's statement as fully truthful unless she herself signals doubt or corrects herself.

Context: {context}

Question under discussion: {question}

Situation 1 -- Lila said: "{statement_1}"
Situation 2 -- Emily said: "{statement_2}"

In which situation is it more probable that the question's claim is true? Reply with only 'situation 1' or 'situation 2', no explanation."""

TASK3_TEMPLATE_E = """Background (accept as given): {context}

Two reliable, honest sources -- Lila and Emily -- were each asked: {question}

Lila: "{statement_1}"
Emily: "{statement_2}"

Given only what each said, which situation makes the claim more probable? Output only 'situation 1' or 'situation 2', nothing else."""
