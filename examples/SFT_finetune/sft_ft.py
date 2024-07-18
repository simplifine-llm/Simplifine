from simplifine_alpha import train_engine

# model name
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# your huggingface token
hf_token = ''

# dataset path/name
dataset_name = 'nlpie/pandemic_pact'

train_engine.hf_sft(model_name, 
        keys=['title', 'abstract', 'explanation'],
        template='''### TITLE: {title}\n ### ABSTRACT: {abstract}\n ###EXPLANATION: {explanation}''',
        response_template='###EXPLANATION:', hf_token=hf_token, zero=True, ddp=False, gradient_accumulation_steps=4, fp16=True, max_seq_length=2048)