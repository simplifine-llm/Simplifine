from eval import evaluation

text = """
In the following years, Rome continued its conquests in Spain with Tiberius Gracchus, and it set foot in Asia, when the last king of Pergamum gave his kingdom to the Roman people. 
The end of the 2nd century brought another threat, when a great host of Germanic peoples, namely Cimbri and Teutones, crossed the river Rhone and moved to Italy. 
Gaius Marius was consul five consecutive times (seven total), and won two decisive battles in 102 and 101 BC. He also reformed the Roman army, giving it such a good reorganisation that it 
remained unchanged for centuries.
The first thirty years of the last century BC were characterised by serious internal problems that threatened the existence of the Republic. The Social War, between Rome and its allies, 
and the Servile Wars (slave uprisings) were hard conflicts,[28] all within Italy, and forced the Romans to change their 
policy with regards to their allies and subjects.[29] By then Rome had become an extensive power, with great wealth which derived from the conquered people (as tribute, food or manpower, i.e. slaves). 
The allies of Rome felt bitter since they had fought by the side of the Romans, and yet they were not citizens and shared little in the rewards. Although they lost the war, 
they finally got what they asked, and by the beginning of the 1st century AD practically all free inhabitants of Italy were Roman citizens.
However, the growth of the Imperium Romanum (Roman power) created new problems, and new demands, that the old political system of the Republic, 
with its annually elected magistrates and its sharing of power, could not solve. Sulla's civil war and his later dictatorship, the extraordinary commands of Pompey Magnus, 
and the first triumvirate made that clear. 
In January 49 BC, Julius Caesar the conqueror of Gaul, crossed the Rubicon with his legions, occupying Rome and beginning a civil war with Pompey. In the following years, 
he vanquished his opponents, and ruled Rome for four years. After his assassination in 44 BC,[30] the Senate tried to reestablish the Republic, but its champions, Marcus Junius Brutus (descendant of the founder of the republic) and Gaius Cassius Longinus were defeated by Caesar's lieutenant Marcus Antonius and Caesar's nephew, Octavian.
"""
summary = """Rome continued its expansion with conquests in Spain under Tiberius Gracchus and received the kingdom of Pergamum in Asia. 
In the late 2nd century BC, Rome faced threats from Germanic tribes, which Gaius Marius notably overcame, also reforming the Roman army. 
The early 1st century BC saw internal conflicts like the Social War and Servile Wars, which led to extended citizenship for all free Italian inhabitants. 
The expansion of Roman power exposed the inadequacies of the Republican political system, evident in conflicts like Sulla's civil war, Pompey's commands, and the rise of the first triumvirate. 
Julius Caesar's crossing of the Rubicon in 49 BC initiated a civil war, leading to his domination of Rome until his assassination in 44 BC. 
Efforts to restore the Republic failed as Caesar's successors defeated republican leaders."""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""

metric_meta_data = {'metric_name': ['RELEVANCY', 'COHERENCE', 'CONSISTENCY', 'FLUENCY'],
                    'metric': [RELEVANCY_SCORE_CRITERIA, COHERENCE_SCORE_CRITERIA, CONSISTENCY_SCORE_CRITERIA, FLUENCY_SCORE_CRITERIA],
                    'metric_scoring': [RELEVANCY_SCORE_STEPS, COHERENCE_SCORE_STEPS, CONSISTENCY_SCORE_STEPS, FLUENCY_SCORE_STEPS]}


eval = evaluation()
eval.summary_eval_batch('bb.json',[text,text], [summary,summary], metric_meta_data)