from simplifine_alpha import train_utils

# model name
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# your huggingface token
hf_token = ''

# dataset path/name
dataset_name = 'nlpie/pandemic_pact'

# your simplifine API key
simplifine_api_key = ''

# and simply, pass it on to us for training!
train_utils.sft_train_cloud(api_key=simplifine_api_key, job_name='sft_cloud_example', model_name=model_name, dataset_name=dataset_name,
                            huggingface_token=hf_token, keys=['title', 'abstract', 'explanation'], 
                            template='''### TITLE: {title}\n ### ABSTRACT: {abstract}\n ###EXPLANATION: {explanation}''',
                            )