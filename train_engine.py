from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss, ContrastiveLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import wandb
import torch.nn.functional as F
from sklearn.metrics import f1_score



def init_device():
    device_name = None
    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
        device_name = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
        device_name = 'cuda'
    else:
        print('Using CPU')
        device = torch.device("cpu")
        device_name = 'cpu'
    return device, device_name


def hf_finetune_llm_qa(model_name:str, dataset_name:str='',
                       from_hf:bool=False, queries:list=[], context:list=[], answers:list=[]):
    """
    Function to fine-tune a language model for question answering
    Args:
    model_name (str): The name of the model to be used
    dataset_name (str): The name of the dataset to be used
    from_hf (bool): A boolean to indicate if the dataset is from huggingface
    queries (list): A list of queries
    context (list): A list of contexts
    answers (list): A list of answers
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device, device_name = init_device()
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if from_hf:
        pass
    else:
        dataset = Dataset.from_dict({'questions':queries, 'contexts':context, 'answers':answers})
        dataset = dataset.train_test_split(0.2)
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['questions'])):
            text = f'''### Question: {example['questions'][i]}\n ### Context: {example['contexts'][i]}, ### Answer: {example['answers'][i]}'''
            output_texts.append(text)
        return output_texts
    
    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    default_args = {
    "output_dir": "tmp",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
    }

    training_args = TrainingArguments(
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    eval_steps=10,
    logging_steps=10,
     **default_args)
    
    trainer = SFTTrainer(
    model,
    max_seq_length=256,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator
    )
    print('\n######################## EVAL PRE-TRAIN ########################\n')
    print(trainer.evaluate())
    print('\n######################## TRAIN ########################\n')
    trainer.train()
    print('\n######################## EVAL POST-TRAIN ########################\n')
    print(trainer.evaluate())



# TODO: attend to this
def hf_finetune_embedder_contrastive(model_name:str, dataset_name:str='',
                         queries:list=[], positives:list=[], negatives:list=[],
                         test_split_ratio:float=0.2,
                         from_hf:bool=False, use_matryoshka:bool=False,
                         matryoshka_dimensions:list=[], 
                         eval_type:str='ir'):
    """
    Function to fine-tune a sentence transformer model using a contrastive loss
    Args:
    model_name (str): The name of the model to be used
    dataset_name (str): The name of the dataset to be used
    queries (list): A list of queries
    positives (list): A list of positive examples
    negatives (list): A list of negative examples
    test_split_ratio (float): The ratio of the dataset to be used for testing
    from_hf (bool): A boolean to indicate if the dataset is from huggingface
    use_matryoshka (bool): A boolean to indicate if matryoshka is to be used
    matryoshka_dimensions (list): A list of dimensions to be used
    eval_type (str): The type of evaluation to be used, accepted: ir, matryoshka, contrastive
    """
    device_name = None
    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
        device_name = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
        device_name = 'cuda'
    else:
        print('Using CPU')
        device = torch.device("cpu")
        device_name = 'cpu'

    # check if the model is available on the GPU, then implemenbt flash attention 2
    if device_name == 'cuda':
        model = SentenceTransformer(
        model_name, 
        model_kwargs={"attn_implementation": "sdpa"}
        ,trust_remote_code=True
        )
    else:
        model = SentenceTransformer(
        model_name, device=device
        ,trust_remote_code=True
        )
    
    if from_hf:
        pass
    else:
        # format should be as follows:
        #     Inputs:
        # +-----------------------------------------------+------------------------------+
        # | Texts                                         | Labels                       |
        # +===============================================+==============================+
        # | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
        # +-----------------------------------------------+------------------------------+
        # example:
        # dataset_train_v2 = Dataset.from_dict({
        #         "sentence1": ["It's nice weather outside today.", "He drove to work."],
        #         "sentence2": ["It's so sunny.", "She walked to the store."],
        #         "label": [1, 0],
        #     })
            
        training_data = {
            "sentence1": queries*2,
            "sentence2": positives + negatives,
            "label": [1]*len(positives) + [0]*len(negatives)
        }
        
        dataset = Dataset.from_dict(training_data).shuffle(seed=22)
        dataset = dataset.train_test_split(test_split_ratio)
        dataset_train = dataset['train']
        dataset_test = dataset['test']
    


    # seperate dics for IR eval
    corpus = {'ids':range(len(positives)+len(negatives)), 'docs':positives+negatives}
    queries = {'ids':range(len(queries)), 'docs':queries}
    queries = dict(zip(queries['ids'], queries['docs']))
    corpus = dict(zip(corpus['ids'], corpus['docs']))

    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    
    evaluator_ir = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="information-retrieval-evaluator",
        score_functions={"cosine": cos_sim},
    )

    if use_matryoshka:
        matryoshka_evaluators = []
        # Iterate over the different dimensions
        for dim in matryoshka_dimensions:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,  # Truncate the embeddings to a certain dimension
                score_functions={"cosine": cos_sim},
            )
            matryoshka_evaluators.append(ir_evaluator)
        evaluator_matryoshka = SequentialEvaluator(matryoshka_evaluators)
        training_loss = MatryoshkaLoss(
    model,  ContrastiveLoss(model), matryoshka_dims=matryoshka_dimensions
    )
    else:
        training_loss = ContrastiveLoss(model)

    if device_name == 'mps':
        optim = 'adamw_torch'
    else:
        optim = 'adamw_torch_fused'

    # define training arguments
    args = SentenceTransformerTrainingArguments(
    output_dir="bge-base-financial-matryoshka", # output directory and hugging face model ID
    num_train_epochs=4,                         # number of epochs
    per_device_train_batch_size=32,             # train batch size
    gradient_accumulation_steps=16,             # for a global batch size of 512
    per_device_eval_batch_size=16,              # evaluation batch size
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",                 # use constant learning rate scheduler
    optim=optim,                  # use fused adamw optimizer
    tf32=False,                                  # use tf32 precision
    bf16=False,                                  # use bf16 precision
    fp16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",                      # evaluate after each epoch
    save_strategy="epoch",                      # save after each epoch
    logging_steps=10,                           # log every 10 steps
    save_total_limit=3,                         # save only the last 3 models
    )
    # load_best_model_at_end=True,                # load the best model when training ends
    # metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
    # )

    
    if use_matryoshka:
        evaluator = evaluator_matryoshka
    else:
        evaluator = evaluator_ir

    trainer = SentenceTransformerTrainer(
    model=model, # bg-base-en-v1
    args=args,  # training arguments
    train_dataset=dataset_train,
    loss=training_loss,
    evaluator=evaluator,
    )

    trainer.train()


# TODO: make the eval type more dynamic
def hf_finetune_embedder_positive(model_name:str, dataset_name:str='',
                         questions:list=[], docs:list=[],
                         from_hf:bool=False, use_matryoshka:bool=False,
                         matryoshka_dimensions:list=[],
                         relevant_ids:list=[],
                         train_split_ratio:float=0.2,
                         do_split:bool=False,):
    print(f'================================================\nQuestions: {len(questions)}')
    device_name = None
    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
        device_name = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
        device_name = 'cuda'
    else:
        print('Using CPU')
        device = torch.device("cpu")
        device_name = 'cpu'

    # check if the model is available on the GPU, then implemenbt flash attention 2
    if torch.cuda.is_available():
        model = SentenceTransformer(
        model_name, 
        model_kwargs={"attn_implementation": "sdpa"}
        ,trust_remote_code=True
        )
    else:
        model = SentenceTransformer(
        model_name, device=device
        ,trust_remote_code=True
        )
    
    if from_hf:
        pass
    else:
        if do_split:
            split_index = int(len(questions)*train_split_ratio)
            questions, questions_eval = questions[:-split_index], questions[-split_index:]
            relevant_ids, relevant_ids_eval = relevant_ids[:-split_index], relevant_ids[-split_index:]

            dataset_q = Dataset.from_dict({'queries':questions_eval})
            dataset_c = Dataset.from_dict({'docs':docs})
            dataset_q = dataset_q.add_column("id", range(len(dataset_q)))
            dataset_c = dataset_c.add_column("id", range(len(dataset_c)))
        else:
            dataset_q = Dataset.from_dict({'queries':questions})
            dataset_c = Dataset.from_dict({'docs':docs})
            dataset_q = dataset_q.add_column("id", range(len(dataset_q)))
            dataset_c = dataset_c.add_column("id", range(len(dataset_c)))


    
        # Convert the datasets to dictionaries
        corpus = dict(
            zip(dataset_c["id"], dataset_c["docs"])
        )  # Our corpus (cid => document)
        queries = dict(
            zip(dataset_q["id"], dataset_q["queries"])
        )  # Our queries (qid => question

        if do_split:
            relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for num,q_id in enumerate(queries):
                relevant_docs[q_id] = [relevant_ids_eval[num]]
        else:
            relevant_docs = {}
            for num,q_id in enumerate(queries):
                relevant_docs[q_id] = [relevant_ids[num]]
    
    evaluator_ir = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="information-retrieval-evaluator",
        score_functions={"cosine": cos_sim},
    )

    if use_matryoshka:
        matryoshka_evaluators = []
        # Iterate over the different dimensions
        for dim in matryoshka_dimensions:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,  # Truncate the embeddings to a certain dimension
                score_functions={"cosine": cos_sim},
            )
            matryoshka_evaluators.append(ir_evaluator)
        evaluator_matryoshka = SequentialEvaluator(matryoshka_evaluators)
        training_loss = MatryoshkaLoss(
    model,  MultipleNegativesRankingLoss(model), matryoshka_dims=matryoshka_dimensions
    )
    else:
        training_loss = MultipleNegativesRankingLoss(model)

    if device_name == 'mps':
        optim = 'adamw_torch'
    else:
        optim = 'adamw_torch_fused'

    # define training arguments
    args = SentenceTransformerTrainingArguments(
    output_dir="bge-base-financial-matryoshka", # output directory and hugging face model ID
    num_train_epochs=4,                         # number of epochs
    per_device_train_batch_size=32,             # train batch size
    gradient_accumulation_steps=16,             # for a global batch size of 512
    per_device_eval_batch_size=16,              # evaluation batch size
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",                 # use constant learning rate scheduler
    optim=optim,                  # use fused adamw optimizer
    tf32=False,                                  # use tf32 precision
    bf16=False,                                  # use bf16 precision
    fp16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",                      # evaluate after each epoch
    save_strategy="epoch",                      # save after each epoch
    logging_steps=10,                           # log every 10 steps
    save_total_limit=3,                         # save only the last 3 models
    )
    # load_best_model_at_end=True,                # load the best model when training ends
    # metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
    # )
    train_dataset = Dataset.from_dict({'anchor':questions, 'positive':[docs[i] for i in relevant_ids]})
    print('\n-----------------------------------------\n')
    print(len(questions))
    print(len(questions_eval))
    print('\n-----------------------------------------\n')
    # eval_dataset = Dataset.from_dict({'anchor':questions_eval, 'positive':[docs[i] for i in relevant_ids_eval]})
    
    if use_matryoshka:
        evaluator = evaluator_matryoshka
    else:
        evaluator = evaluator_ir

    trainer = SentenceTransformerTrainer(
    model=model, # bg-base-en-v1
    args=args,  # training arguments
    train_dataset=train_dataset.select_columns(
        ["positive", "anchor"]
    ),  # training dataset
    loss=training_loss,
    evaluator=evaluator,
    )
    print('\n######################## EVAL PRE-TRAIN ########################\n')
    pre_train_metrics = evaluator(model)
    print(pre_train_metrics)
    trainer.train()
    print('\n######################## EVAL POST-TRAIN ########################\n')
    post_train_metrics = evaluator(model)
    print(post_train_metrics)


# TODO: complete this
def hf_clm_train(model_name:str, dataset_name:str='',
                    context_length:int=128,
                    num_epochs:int=3, batch_size:int=8,
                    lr:float=5e-5, from_hf:bool=True,
                    inputs:list=[], 
                    use_peft:bool=False, peft_config=None,
                    distributed:bool=True, accelerator=None,
                    from_scratch:bool=False):
    
    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
    else:
        print('Using CPU')
        device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    
    if from_hf:
        dataset = load_dataset(dataset_name)
        dataset = dataset['train'][:400]
        dataset = Dataset.from_dict(dataset)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        num_labels = len(set(tokenized_datasets['labels']))

    else:
        tokenized_datasets = Dataset.from_dict({'text':inputs, 'label':labels})
        num_labels = len(set(labels))
        tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    tokenized_datasets = tokenized_datasets.train_test_split(0.2)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)
    if torch.cuda.device_count() > 1 and distributed == True:
        model = torch.nn.DataParallel(model)


def hf_clf_multi_label_train(model_name:str, dataset_name:str='', 
                 num_epochs:int=3, batch_size:int=8, 
                 lr:float=5e-5, from_hf:bool=True,
                 inputs:list=[], labels:list=[],
                 use_peft:bool=False, peft_config=None,
                 accelerator=None, apply_class_weights:bool=False,
                 num_labels:int=0):

    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
        
    else:
        print('Using CPU')
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # assuming labels are already converted to formnat: [1,1,1,0,2,0,0,2,2] and not string
    # TODO: this is not working atm.
    if apply_class_weights:
        label_weights = 1 - labels.sum(axis=0) / labels.sum()
    
    dataset =  Dataset.from_dict({'text': inputs, 'labels': labels}).shuffle(seed=42)
    dataset = dataset.train_test_split(0.2)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            distributed = True
    
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id


    class CustomTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # compute custom loss
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
            return (loss, outputs) if return_outputs else loss
    
    training_args = TrainingArguments(
    output_dir = 'multilabel_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 10,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
    )

    # define which metrics to compute for evaluation
    def compute_metrics(p):
        predictions, labels = p
        f1_micro = f1_score(labels, predictions > 0, average = 'micro')
        f1_macro = f1_score(labels, predictions > 0, average = 'macro')
        f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

    # train
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = small_train_dataset,
        eval_dataset = small_eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
    )

    trainer.train()





def hf_clf_train(model_name:str, dataset_name:str='', 
                 num_epochs:int=3, batch_size:int=8, 
                 lr:float=5e-5, from_hf:bool=True,
                 inputs:list=[], labels:list=[],
                 use_peft:bool=False, peft_config=None,
                 accelerator=None):

    if torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device("cuda")
        
    else:
        print('Using CPU')
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    
    if from_hf:
        dataset = load_dataset(dataset_name)
        dataset = dataset['train'][:10000]
        num_labels = len(set(dataset['label']))
        dataset = Dataset.from_dict(dataset)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        

    else:
        tokenized_datasets = Dataset.from_dict({'text':inputs, 'label':labels})
        num_labels = len(set(labels))
        tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format("torch")
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    tokenized_datasets = tokenized_datasets.train_test_split(0.2)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id




    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.to(device)  
    distributed = False
    if torch.cuda.device_count() > 1:
        distributed = True
        model = torch.nn.DataParallel(model)
    
    if use_peft:
        if peft_config is None:
            peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
        model = get_peft_model(model, peft_config)

    progress_bar = tqdm(range(num_training_steps))

    use_wandb = True
    if use_wandb:
        wandb.init(project="hf_clf_train", config={"model_name": model_name,
                                                   'epochs': num_epochs})
        # wandb.watch(model)

    model.train()
    running_loss=0.0
    if distributed == False:
        for epoch in range(num_epochs):
            for num,batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                running_loss += loss.item()
                 # optional, wandb logging
                if num%50 == 0 and use_wandb:
                    
                    # ---
                    correct = 0
                    total = 0
                    val_loss = 0.0
                    with torch.no_grad():
                        for data in eval_dataloader:
                            labels = data['labels'].to(device)
                            batch = {k: v.to(device) for k, v in data.items()}
                            outputs = model(**batch)
                            predicted = torch.argmax(outputs.logits, dim = -1)
                            total += len(labels)
                            # TODO: this is overhead and will be slow.
                            correct += (predicted == labels).sum().item()
                            val_loss += loss.item()
                    wandb.log({"loss_train": running_loss/50, 'batch': num, 'epoch': epoch, 'accuracy_val': 100 * correct / total, 'loss_val': val_loss/50})
                    running_loss=0.0

    else:
        for epoch in range(num_epochs):
            for num,batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss.mean()
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                running_loss += loss.item()
                 # optional, wandb logging
                if num%50 == 0 and use_wandb:
                    
                    # ---
                    correct = 0
                    total = 0
                    val_loss = 0.0
                    with torch.no_grad():
                        for data in eval_dataloader:
                            labels = data['labels'].to(device)
                            batch = {k: v.to(device) for k, v in data.items()}
                            outputs = model(**batch)
                            predicted = torch.argmax(outputs.logits, dim = -1)
                            total += len(labels)
                            # TODO: this is overhead and will be slow.
                            correct += (predicted == labels).sum().item()
                            val_loss += loss.item()
                    wandb.log({"loss_train": running_loss/50, 'batch': num, 'epoch': epoch, 'accuracy_val': 100 * correct / total, 'loss_val': val_loss/50})
                    running_loss=0.0
                    # ---

if __name__ == '__main__':
    dataset_name = 'yelp_review_full' 
    # model_name = 'google-bert/bert-base-cased'
    model_name = 'openai-community/gpt2'


    hf_clf_multi_label_train(model_name, inputs=['I love this place', 'I hate this place', 'I am neutral about this place'], labels=[[1,0,0],[0,1,0],[1,0,1]], num_labels=3)

    # # model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    # model_name = 'openai-community/gpt2'

    # inputs = ['I love this place', 'I hate this place', 'I am neutral about this place']
    # labels = [0,1,2]
    # hf_clf_train(model_name, dataset_name=dataset_name, from_hf=True, inputs=inputs, labels=labels, use_peft=False)

    # hf_finetune_embedder_positive('sentence-transformers/paraphrase-MiniLM-L6-v2', from_hf=False, queries=['I love this place', 'I hate this place', 'I am neutral about this place'], docs=['I love this place', 'I hate this place', 'I am neutral about this place'],
    #                      use_matryoshka=True, matryoshka_dimensions=[128, 256])
    # hf_finetune_embedder_contrastive('nomic-ai/nomic-embed-text-v1.5', from_hf=False, 
    #                                  queries=['q1', 'q2', 'q3'],
    #                                 positives=['pos1', 'pos2', 'pos3'],
    #                                 negatives=['neg1', 'neg2', 'neg3'],         
    #                                  use_matryoshka=True, matryoshka_dimensions=[128, 256])
#     queries = [
#     'q1: How would you rate your experience with our customer service?',
#     'q2: Are you satisfied with the quality of our product?',
#     'q3: Would you recommend our services to others?',
#     'q4: How do you feel about the pricing of our products?',
#     'q5: Did our website meet your expectations?',
#     'q6: How was the delivery time of your order?',
#     'q7: Is the product easy to use?',
#     'q8: Were you able to find the information you needed on our website?',
#     'q9: How do you feel about the overall design of our website?',
#     'q10: Was our staff helpful during your visit?',
#     'q11: Do you think our product offers good value for money?',
#     'q12: How satisfied are you with our return policy?',
#     'q13: Would you purchase from us again?',
#     'q14: How likely are you to attend one of our events in the future?',
#     'q15: Did you find our mobile app user-friendly?',
#     'q16: How would you rate the quality of our customer support?',
#     'q17: Are you satisfied with the range of products we offer?',
#     'q18: How do you feel about the packaging of your order?',
#     'q19: Did our product meet your expectations?',
#     'q20: How do you feel about the speed of our website?'
# ]

#     positives = [
#         'pos1: Excellent, very satisfied.',
#         'pos2: Yes, absolutely!',
#         'pos3: Definitely, without a doubt.',
#         'pos4: Very reasonable and fair.',
#         'pos5: Yes, it exceeded my expectations.',
#         'pos6: The delivery was very quick.',
#         'pos7: Yes, its very user-friendly.',
#         'pos8: Yes, everything was easy to find.',
#         'pos9: The design is modern and attractive.',
#         'pos10: Yes, they were extremely helpful.',
#         'pos11: Yes, it’s a great value.',
#         'pos12: Very satisfied, it’s very customer-friendly.',
#         'pos13: Absolutely, I would.',
#         'pos14: Very likely, I look forward to it.',
#         'pos15: Yes, it’s very intuitive.',
#         'pos16: Outstanding, very helpful.',
#         'pos17: Yes, there’s a great selection.',
#         'pos18: The packaging was excellent.',
#         'pos19: Yes, it met all my expectations.',
#         'pos20: Very fast and responsive.'
#     ]

#     negatives = [
#         'neg1: Poor, very dissatisfied.',
#         'neg2: No, not at all.',
#         'neg3: No, I wouldn’t.',
#         'neg4: Overpriced and unfair.',
#         'neg5: No, it did not meet my expectations.',
#         'neg6: The delivery was very slow.',
#         'neg7: No, it’s quite difficult to use.',
#         'neg8: No, I couldn’t find what I needed.',
#         'neg9: The design is outdated and unappealing.',
#         'neg10: No, they were not helpful.',
#         'neg11: No, it’s not worth the money.',
#         'neg12: Very dissatisfied, it’s very restrictive.',
#         'neg13: No, I wouldn’t purchase again.',
#         'neg14: Not likely, I’m not interested.',
#         'neg15: No, it’s very confusing.',
#         'neg16: Terrible, not helpful at all.',
#         'neg17: No, the selection is poor.',
#         'neg18: The packaging was terrible.',
#         'neg19: No, it fell short of my expectations.',
#         'neg20: Very slow and unresponsive.'
#     ]
#     hf_finetune_embedder_contrastive('sentence-transformers/paraphrase-MiniLM-L6-v2', from_hf=False, 
#                                      queries=queries,
#                                     positives=positives,
#                                     negatives=negatives,         
#                                      use_matryoshka=True, matryoshka_dimensions=[128, 256])

    # questions = ['q1','q2','q3']
    # contexts = ['c1','c2','c3']
    # answers = ['a1','a2','a3']
    questions = [f'q{i}' for i in range(1000)]
    contexts = [f'c{i}' for i in range(1000)]
    answers = [f'a{i}' for i in range(1000)]
    # from datasets import load_dataset
    # ds = load_dataset("ali77sina/SEC-QA-sorted-chunks")
    # questions = ds['train']['questions'][:100]
    # contexts = ds['train']['sorted_chunks'][:100]
    # answers = ds['train']['answers'][:100]
    # model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    # model_name = 'bigscience/bloom-560m'
    # hf_finetune_llm_qa(model_name, from_hf=False, queries=questions, context=contexts, answers=answers)
    