#%%
from simplifine_alpha import utils
from simplifine_alpha import train_engine

#%%
test_chunks = utils.chunk_text_by_words("This is a sample text to be chunked.", chunk_length=4, overlap=2)
print(test_chunks)
# %%
train_engine.init_device()
# %%
queries = [
    'q1: How would you rate your experience with our customer service?',
    'q2: Are you satisfied with the quality of our product?',
    'q3: Would you recommend our services to others?',
    'q4: How do you feel about the pricing of our products?',
    'q5: Did our website meet your expectations?',
    'q6: How was the delivery time of your order?',
    'q7: Is the product easy to use?',
    'q8: Were you able to find the information you needed on our website?',
    'q9: How do you feel about the overall design of our website?',
    'q10: Was our staff helpful during your visit?',
    'q11: Do you think our product offers good value for money?',
    'q12: How satisfied are you with our return policy?',
    'q13: Would you purchase from us again?',
    'q14: How likely are you to attend one of our events in the future?',
    'q15: Did you find our mobile app user-friendly?',
    'q16: How would you rate the quality of our customer support?',
    'q17: Are you satisfied with the range of products we offer?',
    'q18: How do you feel about the packaging of your order?',
    'q19: Did our product meet your expectations?',
    'q20: How do you feel about the speed of our website?'
]

positives = [
        'pos1: Excellent, very satisfied.',
        'pos2: Yes, absolutely!',
        'pos3: Definitely, without a doubt.',
        'pos4: Very reasonable and fair.',
        'pos5: Yes, it exceeded my expectations.',
        'pos6: The delivery was very quick.',
        'pos7: Yes, its very user-friendly.',
        'pos8: Yes, everything was easy to find.',
        'pos9: The design is modern and attractive.',
        'pos10: Yes, they were extremely helpful.',
        'pos11: Yes, it’s a great value.',
        'pos12: Very satisfied, it’s very customer-friendly.',
        'pos13: Absolutely, I would.',
        'pos14: Very likely, I look forward to it.',
        'pos15: Yes, it’s very intuitive.',
        'pos16: Outstanding, very helpful.',
        'pos17: Yes, there’s a great selection.',
        'pos18: The packaging was excellent.',
        'pos19: Yes, it met all my expectations.',
        'pos20: Very fast and responsive.'
    ]

negatives = [
        'neg1: Poor, very dissatisfied.',
        'neg2: No, not at all.',
        'neg3: No, I wouldn’t.',
        'neg4: Overpriced and unfair.',
        'neg5: No, it did not meet my expectations.',
        'neg6: The delivery was very slow.',
        'neg7: No, it’s quite difficult to use.',
        'neg8: No, I couldn’t find what I needed.',
        'neg9: The design is outdated and unappealing.',
        'neg10: No, they were not helpful.',
        'neg11: No, it’s not worth the money.',
        'neg12: Very dissatisfied, it’s very restrictive.',
        'neg13: No, I wouldn’t purchase again.',
        'neg14: Not likely, I’m not interested.',
        'neg15: No, it’s very confusing.',
        'neg16: Terrible, not helpful at all.',
        'neg17: No, the selection is poor.',
        'neg18: The packaging was terrible.',
        'neg19: No, it fell short of my expectations.',
        'neg20: Very slow and unresponsive.'
    ]

train_engine.hf_finetune_embedder_contrastive('sentence-transformers/paraphrase-MiniLM-L6-v2', from_hf=False, 
                                     queries=queries,
                                    positives=positives,
                                    negatives=negatives,         
                                     use_matryoshka=True, matryoshka_dimensions=[128, 256])
# %%
model_name = 'openai-community/gpt2'

inputs = ['I love this place', 'I hate this place', 'I am neutral about this place']
labels = [0,1,2]
train_engine.hf_clf_train(model_name, dataset_name=dataset_name, from_hf=True, inputs=inputs, labels=labels, use_peft=False)
# %%
