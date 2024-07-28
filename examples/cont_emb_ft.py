from simplifine_alpha import train_engine

# model name
# this should be a sentence transformer model
model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'

# your huggingface token
hf_token = ''

# data
queries = ['the weather is good', 'the weather is bad', 'the weather is okay']
positive = ['nice day', 'bad day', 'okay day']
negative = ['bad day', 'nice day', 'bad day']

train_engine.hf_finetune_embedder_contrastive(model_name=model_name, queries=queries, positive=positive, negative=negative, hf_token=hf_token)
