'''
    Simplfine is an easy-to-use, open-source library for fine-tuning LLMs models quickly on your own hardware or cloud.
    Copyright (C) 2024  Simplifine Corp.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss, ContrastiveLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import wandb
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class wandbConfig:
    wandb_api_key:str
    project:str
    config:dict



def init_device():
    """
    Initializes and returns the appropriate computing device (MPS, CUDA, or CPU) for PyTorch operations.

    This function checks for the availability of Metal Performance Shaders (MPS) on Apple devices,
    then checks for CUDA availability for NVIDIA GPUs, and defaults to CPU if neither is available.

    Returns:
    tuple: A tuple containing the PyTorch device and a string representing the device name.
           - device (torch.device): The selected device for PyTorch operations.
           - device_name (str): The name of the selected device ('mps', 'cuda', or 'cpu').

    Example:
    >>> device, device_name = init_device()
    Using CUDA
    >>> print(device)
    cuda:0
    >>> print(device_name)
    cuda

    Notes:
    - Ensure that the appropriate libraries (e.g., CUDA, MPS) are installed and configured correctly for the device.
    - This function prints the selected device to the console.
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
    return device, device_name


def hf_finetune_llm_qa(model_name:str, dataset_name:str='',
                       from_hf:bool=False, queries:list=[], context:list=[], answers:list=[], hf_token=''):
    """
    Fine-tunes a language model for question answering tasks.

    This function sets up and fine-tunes a language model using the provided dataset or lists of queries, contexts, and answers.
    It supports using datasets from Hugging Face as well as custom datasets.

    Parameters:
    model_name (str): The name of the language model to be used.
    dataset_name (str): The name of the dataset to be used (only if `from_hf` is True).
    from_hf (bool): Indicates if the dataset is from Hugging Face.
    queries (list): A list of query strings for fine-tuning.
    context (list): A list of context strings for fine-tuning.
    answers (list): A list of answer strings for fine-tuning.

    Raises:
    ValueError: If the lengths of queries, context, and answers lists do not match.

    Example:
    >>> hf_finetune_llm_qa(model_name="gpt-2", queries=["What is AI?"], context=["Artificial Intelligence (AI) is ..."], answers=["AI is ..."])

    Notes:
    - This function assumes that `transformers`, `datasets`, and `torch` are installed.
    - It also assumes that the required classes such as `SFTTrainer` and `DataCollatorForCompletionOnlyLM` are defined/imported correctly.
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
    Fine-tunes a sentence transformer model using contrastive loss for embedding tasks.

    This function fine-tunes a given model using contrastive loss with provided queries, positives, and negatives.
    It supports using datasets from Hugging Face or custom datasets.

    Parameters:
    model_name (str): The name of the model to be used.
    dataset_name (str): The name of the dataset to be used (only if `from_hf` is True).
    queries (list): A list of query strings for fine-tuning.
    positives (list): A list of positive example strings.
    negatives (list): A list of negative example strings.
    test_split_ratio (float): The ratio of the dataset to be used for testing.
    from_hf (bool): Indicates if the dataset is from Hugging Face.
    use_matryoshka (bool): Indicates if Matryoshka evaluation is to be used.
    matryoshka_dimensions (list): A list of dimensions to be used for Matryoshka evaluation.
    eval_type (str): The type of evaluation to be used. Accepted values: 'ir', 'matryoshka', 'contrastive'.

    Raises:
    ValueError: If the lengths of queries, positives, and negatives lists do not match.

    Example:
    >>> hf_finetune_embedder_contrastive(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            queries=["What is AI?"], 
            positives=["AI is the simulation of human intelligence processes by machines."], 
            negatives=["The weather is nice today."], 
            test_split_ratio=0.2
        )

    Notes:
    - This function assumes that `transformers`, `datasets`, `sentence_transformers`, and `torch` are installed.
    - It also assumes that the required classes such as `SentenceTransformerTrainer` and `MatryoshkaLoss` are defined/imported correctly.
    - Best format is as follows:
            Inputs:
        +-----------------------------------------------+------------------------------+
        | Texts                                         | Labels                       |
        +===============================================+==============================+
        | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
        +-----------------------------------------------+------------------------------+
        example:
        dataset_train_v2 = Dataset.from_dict({
                "sentence1": ["It's nice weather outside today.", "He drove to work."],
                "sentence2": ["It's so sunny.", "She walked to the store."],
                "label": [1, 0],
            })
    """
    
    device_name = None
    device, deivce_name = init_device()

    # check if the model is available on the GPU, then implement flash attention 2
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
    """
    Fine-tunes a SentenceTransformer model for embedding positive examples.

    Args:
        model_name (str): The name of the model to be used.
        dataset_name (str, optional): Name of the dataset. Defaults to ''.
        questions (list, optional): List of questions. Defaults to [].
        docs (list, optional): List of documents. Defaults to [].
        from_hf (bool, optional): If True, load the model from Hugging Face. Defaults to False.
        use_matryoshka (bool, optional): If True, use matryoshka embeddings. Defaults to False.
        matryoshka_dimensions (list, optional): Dimensions for matryoshka embeddings. Defaults to [].
        relevant_ids (list, optional): List of relevant document IDs. Defaults to [].
        train_split_ratio (float, optional): Ratio for train/test split. Defaults to 0.2.
        do_split (bool, optional): If True, split the data into train and test sets. Defaults to False.

    Returns:
        None
    """
    
    device_name = None
    device, deivce_name = init_device()

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

    train_dataset = Dataset.from_dict({'anchor':questions, 'positive':[docs[i] for i in relevant_ids]})
    print('\n-----------------------------------------\n')
    print(len(questions))
    print(len(questions_eval))
    print('\n-----------------------------------------\n')
    
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

    
def cleanup():
    """
    function to cleanup the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class PromptConfig:
    new_padding_token: Optional[str] = None
    use_chat_template: Optional[bool] = False
    system_message_key: Optional[str] = None
    system_message: Optional[str] = None
    clm_column: Optional[str] = None
    context_length: Optional[int] = 1024


@dataclass
class sftPromptConfig(PromptConfig):
    keys: List[str] = field(default_factory=list)
    template: str = ""
    response_template: str = ""


def sft_train(
    model_name:str, dataset_name:str=None, hf_token:str='', dataset_config_name:str=None, data_from_hf:bool=True,
    do_split:bool=True, split_ratio:float=0.2, use_peft:bool=False, lora_config:LoraConfig=None, 
    sft_config:SFTConfig=None, data:dict={}, wandb_config:wandbConfig=None, 
    use_ddp:bool=False, use_zero:bool=True, sft_prompt_config:sftPromptConfig=None
): 
    """
    Train a model using the Supervised Finetuning (SFT) process.

    This function initializes a model, tokenizer, and trainer for the SFT process using Hugging Face Transformers.

    Parameters:
    -----------
    model_name : str
        The name or path of the pre-trained model to use.
    dataset_name : str, optional
        The name of the dataset to use for training.
    keys : list, optional
        List of keys used for formatting prompts in the template.
    template : str, optional
        Template string for formatting prompts with keys.
    do_split : bool, optional
        Whether to split the dataset into training and testing sets. Default is True.
    split_ratio : float, optional
        The ratio of the dataset to use for testing. Default is 0.2.
    sft_config : SFTConfig, optional
        Configuration object for the SFT process. Default is None.
    data : dict, optional
        Dictionary containing the dataset for training. Default is an empty dictionary.
    num_epochs : int, optional
        Number of training epochs. Default is 3.
    batch_size : int, optional
        Batch size for training. Default is 1.
    wandb_config : wandbConfig, optional
        Configuration object for Weights & Biases. Default is None.
    use_ddp : bool, optional
        Whether to use Distributed Data Parallel (DDP). Default is False.
    use_zero : bool, optional
        Whether to use Zero optimization. Default is True.

    Raises:
    -------
    ValueError
        If both DDP and Zero optimization are set to True simultaneously. Only one dist method is accepted at once.

    Notes:
    ------
    - This function initializes a tokenizer, configures SFT parameters, loads the dataset, initializes the model, and starts training using the SFTTrainer.
    - If DDP and Zero optimization are enabled, they cannot be used simultaneously due to conflicting configurations.
    """
    # parse the sft prompt config
    new_padding_token = sft_prompt_config.new_padding_token
    keys = sft_prompt_config.keys
    template = sft_prompt_config.template
    response_template = sft_prompt_config.response_template
    use_chat_template = sft_prompt_config.use_chat_template
    system_message_key = sft_prompt_config.system_message_key
    system_message = sft_prompt_config.system_message
    # Ensure no default process group exists
    if dist.is_initialized():
        print("Destroying existing process group")
        dist.destroy_process_group()
    
    if use_ddp and use_zero:
        raise ValueError("Only one dist method is accepted at once.")
    
    if use_ddp:
        sft_config.deepspeed = None
    
    if response_template not in template:
        raise ValueError('The response template must be in the template')

    if system_message_key and system_message:
        raise ValueError('Only provide key from dataset or system message as a string, not both')
    
    if sft_config is None:
        raise ValueError('SFT config must be provided')
    
    if data_from_hf and not dataset_name:
        raise ValueError('Dataset name must be provided if data is from Hugging Face')
    
    # initialize the training arguments
    if sft_config is None:
        raise ValueError('SFT config must be provided')
    
    if use_ddp and sft_config.gradient_checkpointing:
        print('[WARNING]: Gradient checkpointing is not supported with DDP. Disabling gradient checkpointing.')
        sft_config.gradient_checkpointing = False

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
    if tokenizer.pad_token is None:
        if new_padding_token:
            tokenizer.pad_token = new_padding_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # replacing the response template, with chat/instruction based tokens.
    # tokenizing response tempaltes in context can be different to ones without it
    if use_chat_template:
            if tokenizer.chat_template:
                # dummy messages to extract chat tokens
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Who won the world series in 2020?"
                    },
                    {
                        "role": "assistant",
                        "content": "The Los Angeles Dodgers won the World Series in 2020."
                    }
            ]   
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                chat_temp = text.split(messages[1]['content'])[-1].split(messages[-1]['content'])[0]
                chat_response_temp = chat_temp.replace(tokenizer.eos_token,'')
            else:
                raise ValueError('Tokenizer does not have chat template')
    
    sft_config.remove_unused_columns=False
    if data_from_hf:
        try:
            if dataset_config_name:
                raw_datasets = load_dataset(dataset_name, dataset_config_name, token=False, split='train')
            else:
                raw_datasets = load_dataset(dataset_name, token=False, split='train')
        except Exception as e:
            print(f'Error: {e}')
            if dataset_config_name:
                raw_datasets = load_dataset(dataset_name, dataset_config_name, token=False)
            else:
                raw_datasets = load_dataset(dataset_name, token=False)
            raw_datasets = raw_datasets['train']
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)
    else:
        raw_datasets = Dataset.from_dict(data)
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)
    
    # function to format the prompts
    def formatting_prompts_func(example):
        output_texts = []
            
        if not tokenizer.chat_template:
                for i in range(len(example[keys[0]])):
                    formatted_text = template.format(
                        **{key: example[key][i] for key in keys}
                    )
                    output_texts.append(formatted_text)
        else:
            for i in range(len(example[keys[0]])):
                formatted_text = template.format(
                    **{key: example[key][i] for key in keys}
                )
                user_text, assistant_text = formatted_text.split(response_template)
                assistant_text = response_template + assistant_text
                if system_message_key or system_message:
                    if system_message_key:
                        system_message_text = example[system_message_key][i]
                    else:
                        system_message_text = system_message
                    messages = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                        {"role": "system", "content": system_message_text}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(chat_prompt)
        
        if not all(isinstance(text, str) for text in output_texts):
            raise ValueError("Formatted text must be a list of strings")
    
        # Ensure that there's at least one text to process
        if not output_texts:
            return {'input_ids': [], 'attention_mask': []}
    
        tokenized_output = tokenizer(output_texts, truncation=True, padding='max_length', add_special_tokens=True, max_length=tokenizer.model_max_length)

        return tokenized_output

    if tokenizer.chat_template:
        collator = DataCollatorForCompletionOnlyLM(chat_response_temp, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
 
    promptTokenizedDataset = raw_datasets.map(formatting_prompts_func, batched=True, remove_columns=raw_datasets['train'].column_names)
    promptTokenizedDataset = promptTokenizedDataset.shuffle(len(promptTokenizedDataset))

    # initialize the peft config
    if use_peft:
        if lora_config is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
    
    # initialize the model
    device, device_name = init_device()
    if device_name == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token)
    model.resize_token_embeddings(len(tokenizer))

    # if peft is enabled, use the peft model
    if use_peft:
        model = get_peft_model(model, peft_config=lora_config)
    
    if sft_config.deepspeed is None and use_zero:
        raise ValueError('Zero optimization requires a DeepSpeed config')
    
    _is_distritubed = False
    if torch.cuda.device_count() > 1:
        if use_ddp and use_zero:
            raise ValueError('Zero optimization and DDP cannot be used together')
        if use_ddp:
            _is_distritubed = True
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")

            sft_config.ddp_find_unused_parameters=False
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
        elif use_zero:
            _is_distritubed = True
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")
            print('using ZeRO optimization')
        else:
            # user warning not utilizing DDP or ZeRO for multi-gpu setup
            print('[WARNING]: multiple GPUs detected, but not using DDP or ZeRO')
    else:
        model.to(device)
        distributed = False
        if device_name == 'mps':
            sft_config.fp16 = False
            sft_config.bf16 = False
        
    print(f"train data set is: {promptTokenizedDataset['train']}, eval dataset is {promptTokenizedDataset['test']}")
    if do_split:
        trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=promptTokenizedDataset['train'],
        eval_dataset=promptTokenizedDataset['test'],
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator
        )
    else:
        trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=promptTokenizedDataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator
        )

    # creating a directory in ouput dir for final model saving
    output_dir_final = os.path.join(sft_config.output_dir, 'final_model')
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final, exist_ok=True)

    trainer.train()
    trainer.save_model(output_dir_final)

    if _is_distritubed:
        dist.destroy_process_group()
    

def hf_sft(model_name:str, dataset_name:str='nlpie/pandemic_pact',
           keys:list=[], template:str='', do_split:bool=True, split_ratio:float=0.2, load_eval_from_data:bool=False, 
           data:dict={}, num_epochs:int=3, batch_size:int=1, wandb_api_key:str='',
           lr:float=5e-5, from_hf:bool=True, response_template:str='### Answer:', eval_steps:int=10, logging_steps:int=10,
           use_peft:bool=False, peft_config=None, ddp:bool=False, zero:bool=True, deepspeed_config:str='home/ubuntu/src/zero_config.json',
           hf_token:str='', gradient_accumulation_steps:int=1, fp16:bool=False, bf16:bool=False, report_to:str='none',
            gradient_checkpointing:bool=False, max_seq_length:int=2048, use_wandb:bool=False, output_dir:str='sft_output', eval_accumulation_steps:int=8, wandb_config:wandbConfig=None):
    
    """
    Execute the SFT (Supervised Finetuning) process using Hugging Face Transformers.

    This function initializes a model, tokenizer, and trainer for Supervised Finetuning tasks using Hugging Face Transformers.

    Parameters:
    -----------
    model_name : str
        The name or path of the pre-trained model to use.
    dataset_name : str, optional
        The name of the dataset to use for training. Default is 'ali77sina/SEC-QA-sorted-chunks'.
    keys : list, optional
        List of keys used for formatting prompts in the template.
    template : str, optional
        Template string for formatting prompts with keys.
    string_data : list, optional
        List of strings to process as data.
    num_epochs : int, optional
        Number of training epochs. Default is 3.
    batch_size : int, optional
        Batch size for training. Default is 2.
    lr : float, optional
        Learning rate for optimization. Default is 5e-5.
    from_hf : bool, optional
        Whether to load dataset from Hugging Face datasets. Default is True.
    response_template : str, optional
        Template string for response formatting. Default is '### Answer:'.
    use_peft : bool, optional
        Whether to use PEFT (Parameterized Fine-Tuning). Default is False.
    peft_config : object, optional
        Configuration object for PEFT. Default is None.
    ddp : bool, optional
        Whether to use Distributed Data Parallel (DDP). Default is False.
    zero : bool, optional
        Whether to use Zero optimization. Default is True.

    Raises:
    -------
    ValueError
        If both DDP and Zero optimization are set to True simultaneously. only one dist method is accepted at once.

    Notes:
    ------
    - This function initializes a tokenizer, configures SFT parameters, loads the dataset, initializes the model, and starts training using the SFTTrainer.
    - If DDP and Zero optimization are enabled, they cannot be used simultaneously due to conflicting configurations.
    """
    
    # Ensure no default process group exists
    if dist.is_initialized():
        print("Destroying existing process group")
        dist.destroy_process_group()
    
    # deepspeed config, fix this for the time being
    deepspeed_config='home/ubuntu/src/zero_config.json'
    
    if response_template not in template:
        raise ValueError('The response template must be in the template')
    
    # script_path = os.path.dirname(os.path.realpath(__file__))
    # output_dir = os.path.join(script_path, output_dir)
    
    # init wandb
    if report_to == 'wandb':
        wandb_api_key = wandb_config.api_key
        project = wandb_config.project
        config = wandb_config.config
        wandb.login(key=wandb_api_key)
        wandb.init(project=project, config=config)
    
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # replacing the response template, with chat/instruction based tokens.
    # tokenizing response tempaltes in context can be different to ones without it
    if tokenizer.chat_template:
        # dummy messages to extract chat tokens
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Who won the world series in 2020?"
            },
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020."
            }
    ]   
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        chat_temp = text.split(messages[1]['content'])[-1].split(messages[-1]['content'])[0]
        chat_response_temp = chat_temp.replace(tokenizer.eos_token,'')


    if from_hf:
        try:
            raw_datasets = load_dataset(dataset_name, token=False, split='train')
        except Exception as e:
            print(f'Error: {e}')
            raw_datasets = load_dataset(dataset_name, token=False)
            raw_datasets = raw_datasets['train']
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)
    else:
        raw_datasets = Dataset.from_dict(data)
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)

    def formatting_prompts_func(example):
        output_texts = []
            
        if not tokenizer.chat_template:
                for i in range(len(example[keys[0]])):
                    formatted_text = template.format(
                        **{key: example[key][i] for key in keys}
                    )
                    output_texts.append(formatted_text)
        else:
            for i in range(len(example[keys[0]])):
                formatted_text = template.format(
                    **{key: example[key][i] for key in keys}
                )
                user_text, assistant_text = formatted_text.split(response_template)
                assistant_text = response_template + assistant_text
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                output_texts.append(chat_prompt)
        
        if not all(isinstance(text, str) for text in output_texts):
            raise ValueError("Formatted text must be a list of strings")
    
        # Ensure that there's at least one text to process
        if not output_texts:
            return {'input_ids': [], 'attention_mask': []}
    
        tokenized_output = tokenizer(output_texts, truncation=True, padding='max_length', add_special_tokens=True)

        return tokenized_output

    if tokenizer.chat_template:
        collator = DataCollatorForCompletionOnlyLM(chat_response_temp, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
 
    promptTokenizedDataset = raw_datasets.map(formatting_prompts_func, batched=True, remove_columns=raw_datasets['train'].column_names)
    promptTokenizedDataset = promptTokenizedDataset.shuffle(len(promptTokenizedDataset))

    # initialize the peft config
    if use_peft:
        if not peft_config:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
    
    # initialize the model
    device, device_name = init_device()
    if device_name == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token,
                                                 torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # if peft is enabled, use the peft model
    if use_peft:
        model = get_peft_model(model, peft_config=peft_config)

    
    if torch.cuda.device_count() > 1:
        if ddp and zero:
            raise ValueError('Zero optimization and DDP cannot be used together')
        if ddp:
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")
                
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
            distributed = True
            sft_config = SFTConfig(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            max_seq_length = max_seq_length,
                            report_to=report_to,
                            remove_unused_columns=False,
                            eval_steps=eval_steps,
                            eval_accumulation_steps=eval_accumulation_steps,
                            evaluation_strategy="steps",
                            logging_steps=logging_steps
                                    )
        elif zero:
            sft_config = SFTConfig(
                            output_dir="/home/ubuntu/src//tmp",
                            deepspeed="/home/ubuntu/src/zero_config.json",
                            per_device_train_batch_size = 1,
                            per_device_eval_batch_size = 1,
                            num_train_epochs= 1,
                            fp16=True,
                            learning_rate=2e-5,
                            gradient_accumulation_steps=4,
                            report_to='none',
                            gradient_checkpointing=True,
                            logging_dir="/home/ubuntu/src//chkp-pact",
                            max_seq_length = max_seq_length,
                            save_steps=50,
                            eval_steps=1,
                            eval_accumulation_steps=eval_accumulation_steps,
                            evaluation_strategy="steps",
                            logging_steps=logging_steps)
        else:
            sft_config = SFTConfig(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            max_seq_length = max_seq_length,
                            report_to=report_to,
                            eval_steps=eval_steps,
                            eval_accumulation_steps=eval_accumulation_steps,
                            logging_steps=logging_steps,
                            evaluation_strategy="steps"
                                    )
    else:
        model.to(device)
        distributed = False
        if device_name == 'mps':
            fp16 = False
        sft_config = SFTConfig(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=False,
                            bf16=False,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            max_seq_length = max_seq_length,
                            report_to=report_to,
                            remove_unused_columns=True,
                            eval_steps=eval_steps,
                            eval_accumulation_steps=eval_accumulation_steps,
                            logging_steps=logging_steps,
                            evaluation_strategy="steps"
                                    )
    
    print(f"train data set is: {promptTokenizedDataset['train']}, eval dataset is {promptTokenizedDataset['test']}")
    if do_split:
        trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=promptTokenizedDataset['train'],
        eval_dataset=promptTokenizedDataset['test'],
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator
        )
    else:
        trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=promptTokenizedDataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator
        )

    # creating a directory in ouput dir for final model saving
    output_dir_final = os.path.join(output_dir, 'final_model')
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final, exist_ok=True)


    trainer.train()
    trainer.save_model(output_dir_final)

    if ddp:
        dist.destroy_process_group()

def clm_train(
    model_name:str, dataset_name:str=None, hf_token:str='', dataset_config_name:str=None, data_from_hf:bool=True,
    do_split:bool=True, split_ratio:float=0.2, use_peft:bool=False, lora_config:LoraConfig=None, 
    train_args:TrainingArguments=None, data:dict={}, wandb_config:wandbConfig=None, 
    use_ddp:bool=False, use_zero:bool=True, prompt_config:PromptConfig=None
    ):
    # Ensure no default process group exists
    if dist.is_initialized():
        print("Destroying existing process group")
        dist.destroy_process_group()
    
    if use_ddp and use_zero:
        raise ValueError("Only one dist method is accepted at once.")
    
    if use_ddp:
        train_args.deepspeed = None

    if train_args is None:
        raise ValueError('SFT config must be provided')
    
    if data_from_hf and not dataset_name:
        raise ValueError('Dataset name must be provided if data is from Hugging Face')
    
    # initialize the training arguments
    if train_args is None:
        raise ValueError('SFT config must be provided')
    
    if use_ddp and train_args.gradient_checkpointing:
        print('[WARNING]: Gradient checkpointing is not supported with DDP. Disabling gradient checkpointing.')
        train_args.gradient_checkpointing = False

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_token)
    if tokenizer.pad_token is None:
        if prompt_config.new_padding_token:
            tokenizer.pad_token = prompt_config.new_padding_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    train_args.remove_unused_columns=False
    if data_from_hf:
        try:
            if dataset_config_name:
                raw_datasets = load_dataset(dataset_name, dataset_config_name, token=False, split='train')
                raw_datasets = raw_datasets.rename_column(prompt_config.clm_column, 'text')
            else:
                raw_datasets = load_dataset(dataset_name, token=False, split='train')
                raw_datasets = raw_datasets.rename_column(prompt_config.clm_column, 'text')
        except Exception as e:
            print(f'Error: {e}')
            if dataset_config_name:
                raw_datasets = load_dataset(dataset_name, dataset_config_name, token=False)
            else:
                raw_datasets = load_dataset(dataset_name, token=False)
            raw_datasets = raw_datasets['train']
            raw_datasets = raw_datasets.rename_column(prompt_config.clm_column, 'text')
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)
    else:
        raw_datasets = Dataset.from_dict(data)
        if do_split:
            raw_datasets = raw_datasets.train_test_split(split_ratio)
    
    # tokenizing the dataset
    def tokenize(element):
        outputs = tokenizer(
            element['text'],
            truncation=True,
            max_length=prompt_config.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == prompt_config.context_length:
                input_batch.append(input_ids)

        return {"input_ids": input_batch}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets['train'].column_names
    )
    
    if train_args.report_to == ['wandb']:
        wandb_api_key = wandb_config.api_key
        project = wandb_config.project
        config = wandb_config.config
        wandb.login(key=wandb_api_key)
        wandb.init(project=project, config=config)


    # initialize the peft config
    if use_peft:
        if lora_config is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
    
    # initialize the model
    device, device_name = init_device()
    if device_name == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 token = hf_token)
    model.resize_token_embeddings(len(tokenizer))

    # if peft is enabled, use the peft model
    if use_peft:
        model = get_peft_model(model, peft_config=lora_config)
    
    _is_distritubed = False
    if torch.cuda.device_count() > 1:
        if use_ddp and use_zero:
            raise ValueError('Zero optimization and DDP cannot be used together')
        if use_ddp:
            _is_distritubed = True
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")

            train_args.ddp_find_unused_parameters=False
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
        elif use_zero:
            _is_distritubed = True
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")
            print('using ZeRO optimization')
        else:
            # user warning not utilizing DDP or ZeRO for multi-gpu setup
            print('[WARNING]: multiple GPUs detected, but not using DDP or ZeRO')
    else:
        model.to(device)
        distributed = False
        if device_name == 'mps':
            train_args.fp16 = False
            train_args.bf16 = False
    
    # init training
    if do_split:
        trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        )
    else:
        trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
        )

    # creating a directory in ouput dir for final model saving
    output_dir_final = os.path.join(train_args.output_dir, 'final_model')
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final, exist_ok=True)
    
    trainer.train()
    trainer.save_model(output_dir_final)
    
    if use_ddp:
      dist.destroy_process_group()




def hf_clm_train(model_name:str='', dataset_name:str="",
                    context_length:int=128, data:list=[],
                    num_epochs:int=3, batch_size:int=8, fp16:bool=False, bf16:bool=False,
                    lr:float=5e-5, from_hf:bool=True, do_split:bool=True, split_ratio:float=0.2,
                    gradient_accumulation_steps:int=4, gradient_checkpointing:bool=False,
                    report_to:str='none', wandb_api_key:str='', wandb_config:wandbConfig=None,
                    use_peft:bool=False, peft_config=None, hf_token:str='',
                    hf_column:str='text', lr_scheduler_type:str='linear', eval_accumulation_steps:int=8,
                    output_dir:str='clm_output', ddp:bool=False, zero:bool=True):
    
    """
    Train a causal language model using Hugging Face Transformers.

    Parameters:
    -----------
    model_name : str
        The name or path of the pre-trained model to use.
    dataset_name : str, optional
        The name of the dataset to use for training. 
    context_length : int, optional
        Maximum length of the input sequences. Default is 128.
    string_data : List[str], optional
        List of strings to use as data if `from_hf` is False. Default is an empty list.
    num_epochs : int, optional
        Number of training epochs. Default is 3.
    batch_size : int, optional
        Batch size for training. Default is 8.
    lr : float, optional
        Learning rate for optimization. Default is 5e-5.
    from_hf : bool, optional
        Whether to load dataset from Hugging Face datasets. Default is True.
    inputs : List[str], optional
        List of inputs. Default is an empty list.
    use_peft : bool, optional
        Whether to use PEFT (Parameterized Fine-Tuning). Default is False.
    peft_config : object, optional
        Configuration object for PEFT. Default is None.
    distributed : bool, optional
        Whether to use distributed training. Default is True.
    accelerator : object, optional
        Accelerator object for distributed training. Default is None.
    from_scratch : bool, optional
        Whether to start training from scratch. Default is False.
    train_test_split_ratio : float, optional
        Ratio to split training and testing datasets. Default is 0.2.
    output_dir : str, optional
        Directory to save model outputs. Default is 'clm_output'.
    ddp : bool, optional
        Whether to use Distributed Data Parallel (DDP). Default is False.
    zero : bool, optional
        Whether to use Zero optimization. Default is True.

    Raises:
    -------
    ValueError
        If both DDP and Zero optimization are set to True simultaneously.

    Notes:
    ------
    - This function initializes a tokenizer, configures training arguments, loads the dataset, initializes the model,
      and starts training using the Trainer from Hugging Face Transformers.
    - It supports distributed training using DDP or zero optimization if configured.
    """

    # Ensure no default process group exists
    if dist.is_initialized():
        print("Destroying existing process group")
        dist.destroy_process_group()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # init wandb
    if report_to == 'wandb':
        wandb_api_key = wandb_config.api_key
        project = wandb_config.project
        config = wandb_config.config
        wandb.login(key=wandb_api_key)
        wandb.init(project=project, config=config)

    # intialize the device  
    device, deivce_name = init_device()

    # initialize the peft config
    if use_peft:
        if not peft_config:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )

    # load the dataset
    if not from_hf: 
        raw_dataset = Dataset.from_dict({'text': data})
        if do_split:
            raw_dataset = raw_dataset.train_test_split(split_ratio)
    else:
        try:
            raw_dataset = load_dataset(dataset_name, token=False, split='train')
        except Exception as e:
            print(f'Error: {e}, this is because the split parameter is not available')
            raw_dataset = load_dataset(dataset_name, token=False)
        raw_dataset = raw_dataset.rename_column(hf_column, 'text')
        if do_split:
            raw_dataset = raw_dataset.train_test_split(split_ratio)
    

    # initialize the model
    model = AutoModelForCausalLM.from_pretrained(model_name, token = hf_token)
    model.resize_token_embeddings(len(tokenizer))

    # if peft is enabled, use the peft model
    if use_peft:
        model = get_peft_model(model, peft_config=peft_config)

    device, device_name = init_device()
    if torch.cuda.device_count() > 1:
        if ddp and zero:
            raise ValueError('Zero optimization and DDP cannot be used together')
        
        if ddp:
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")
                
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
            distributed = True
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                            lr_scheduler_type=lr_scheduler_type,
                            eval_accumulation_steps=eval_accumulation_steps
                                    )
        elif zero:
            # sft_config = SFTConfig(
            #                 output_dir=output_dir,
            #                 deepspeed='/home/ubuntu/src/zero_config.json',
            #                 per_device_train_batch_size = batch_size,
            #                 per_device_eval_batch_size = batch_size,
            #                 num_train_epochs= num_epochs,
            #                 fp16=fp16,
            #                 bf16=bf16,
            #                 learning_rate=lr,
            #                 gradient_accumulation_steps=gradient_accumulation_steps,
            #                 gradient_checkpointing=gradient_checkpointing,
            #                 max_seq_length = max_seq_length,
            #                 report_to=report_to
            #                         )
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            deepspeed="/home/ubuntu/src/zero_config.json",
                            per_device_train_batch_size = 1,
                            per_device_eval_batch_size = 1,
                            num_train_epochs= 1,
                            fp16=True,
                            learning_rate=2e-5,
                            gradient_accumulation_steps=4,
                            report_to='none',
                            gradient_checkpointing=True,
                            remove_unused_columns=False,
                            logging_steps=20,
                            save_steps=50,
                            eval_steps=1,
                            lr_scheduler_type=lr_scheduler_type,
                            eval_accumulation_steps=eval_accumulation_steps
                            )
        else:
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                            lr_scheduler_type=lr_scheduler_type,
                            eval_accumulation_steps=eval_accumulation_steps
                                    )
    else:
        model.to(device)
        distributed = False
        if device_name == 'mps':
            fp16 = False
            bf16 = False
        TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                            lr_scheduler_type=lr_scheduler_type,
                            eval_accumulation_steps=eval_accumulation_steps
                                    )


    def tokenize(element):
        outputs = tokenizer(
            element['text'],
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
    )


    trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainArgs,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    )

    # creating a directory in ouput dir for final model saving
    output_dir_final = os.path.join(output_dir, 'final_model')
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final, exist_ok=True)
    
    trainer.train()
    trainer.save_model(output_dir_final)
    
    if ddp:
      dist.destroy_process_group()



def hf_clf_multi_label_train(model_name:str, dataset_name:str='', 
                 num_epochs:int=3, batch_size:int=8, 
                 lr:float=5e-5, from_hf:bool=True,
                 inputs:list=[], labels:list=[],
                 use_peft:bool=False, peft_config=None,
                 accelerator=None, apply_class_weights:bool=False,
                 num_labels:int=0):

    device, deivce_name = init_device()

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
    load_best_model_at_end = True,
    report_to='none'
    )

    # define which metrics to compute for evaluation
    def compute_metrics(p):
        """
    Compute F1 scores for evaluation.

    Parameters:
    -----------
    p : tuple
        Tuple containing predictions and labels.

    Returns:
    --------
    dict
        Dictionary containing computed F1 scores:
        - 'f1_micro': Micro-average F1 score.
        - 'f1_macro': Macro-average F1 score.
        - 'f1_weighted': Weighted-average F1 score.
    """
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


def hf_clf_train(model_name:str, dataset_name:str='', hf_data_column:str='', hf_label_column:str='',
                 num_epochs:int=3, batch_size:int=8, 
                 lr:float=5e-5, from_hf:bool=True, hf_token:str='',
                 inputs:list=[], labels:list=[], output_dir:str='clf_output',
                 use_peft:bool=False, peft_config=None, 
                 report_to='none', wandb_api_key:str='',
                 ddp:bool=False, zero:bool=False, fp16:bool=False, bf16:bool=False,
                 gradient_accumulation_steps:int=1, gradient_checkpointing:bool=False):

    """
Train a sequence classification model using Hugging Face transformers.

This function trains a sequence classification model on a given dataset using
the specified Hugging Face model. It supports GPU acceleration, optional
PEFT (Performance Efficient Transfer) model integration, and optional logging
with Weights & Biases (WandB). The function handles dataset loading, tokenization,
model initialization, optimizer setup, and training loop execution.

Parameters:
-----------
model_name : str
    The name or path of the pretrained Hugging Face model to use.
dataset_name : str, optional
    Name of the dataset to load from the Hugging Face datasets library.
    Default is an empty string, indicating that the dataset is provided 
    through inputs and labels directly.
num_epochs : int
    Number of epochs to train the model.
batch_size : int
    Batch size for training and evaluation.
lr : float
    Learning rate for the optimizer.
from_hf : bool, optional
    Whether to load the dataset from Hugging Face datasets library.
    Default is True.
inputs : list, optional
    List of input texts if loading dataset not from Hugging Face.
labels : list, optional
    List of labels corresponding to inputs if loading dataset not from Hugging Face.
use_peft : bool, optional
    Whether to apply Performance Efficient Transfer (PEFT) model.
peft_config : dict, optional
    Configuration dictionary for PEFT model setup.
accelerator : object, optional
    Accelerator object for distributed training.
use_wandb : bool, optional
    Whether to log training progress using Weights & Biases (WandB).
    Default is False.

Returns:
--------
None

Raises:
-------
None

Examples:
---------
# Example usage with a pretrained model from Hugging Face datasets library
hf_clf_train(model_name="bert-base-uncased", dataset_name="glue", num_epochs=3,
             batch_size=8, lr=5e-5, from_hf=True)

# Example usage with custom inputs and labels
inputs = ["Sample input 1", "Sample input 2"]
labels = [0, 1]
hf_clf_train(model_name="bert-base-uncased", inputs=inputs, labels=labels,
             num_epochs=3, batch_size=8, lr=5e-5, from_hf=False)

Notes:
------
- Ensure the Hugging Face model specified (`model_name`) supports sequence classification.
- This function supports GPU acceleration and distributed training if multiple GPUs are available.
- PEFT (Performance Efficient Transfer) can be enabled for optimizing model performance.
- WandB integration (`use_wandb=True`) enables logging of training metrics and progress.
"""

    # Ensure no default process group exists
    if dist.is_initialized():
        print("Destroying existing process group")
        dist.destroy_process_group()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # init wandb
    if report_to == 'wandb':
        wandb.login(key=wandb_api_key)
        wandb.init(project="clm_train", config={"model_name": model_name,
                                                   'epochs': num_epochs})

    device, deivce_name = init_device()
    

    def tokenize_function(examples):
        try:
            texts = examples["text"]
            # Ensure texts is a list of strings
            if not isinstance(texts, list):
                texts = [texts]
            return tokenizer(texts, padding="max_length", truncation=True)
        except Exception as e:
            print(f"Error during tokenization: {e}")
            print(f"Input data: {examples['text']}")
            raise
    
    
    if from_hf:
        dataset = load_dataset(dataset_name)
        # load the train split if exists
        if 'train' in dataset:
            dataset = dataset['train']
        inputs = dataset[hf_data_column]
        labels = dataset[hf_label_column]
        tokenized_datasets = Dataset.from_dict({'text':inputs, 'label':labels})
    else:
        tokenized_datasets = Dataset.from_dict({'text':inputs, 'label':labels})

    print(tokenized_datasets)
    num_labels = len(set(labels))
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    tokenized_datasets = tokenized_datasets.train_test_split(0.2)

    print(tokenized_datasets)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, token = hf_token)
    model.config.pad_token_id = model.config.eos_token_id


    model.to(device)  
    distributed = False
    # if peft is enabled, use the peft model
    if use_peft:
        if not peft_config:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, peft_config=peft_config)
        

    device, device_name = init_device()
    if torch.cuda.device_count() > 1:
        if ddp and zero:
            raise ValueError('Zero optimization and DDP cannot be used together')
        
        if ddp:
            if not dist.is_initialized():
                print("Initializing process group for DDP")
                dist.init_process_group("nccl", world_size=torch.cuda.device_count())
            else:
                print("Process group already initialized")
                
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
            distributed = True
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                                    )
        elif zero:
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            deepspeed="/home/ubuntu/src/zero_config.json",
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            report_to=report_to,
                            gradient_checkpointing=gradient_checkpointing,
                            remove_unused_columns=False,
                            )
        else:
            TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                                    )
    else:
        model.to(device)
        distributed = False
        if device_name == 'mps':
            fp16 = False
            bf16 = False
        TrainArgs = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size = batch_size,
                            per_device_eval_batch_size = batch_size,
                            num_train_epochs= num_epochs,
                            fp16=fp16,
                            bf16=bf16,
                            learning_rate=lr,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            gradient_checkpointing=gradient_checkpointing,
                            report_to=report_to,
                            remove_unused_columns=False,
                                    )
    
    trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainArgs,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    )

    # creating a directory in ouput dir for final model saving
    output_dir_final = os.path.join(output_dir, 'final_model')
    if not os.path.exists(output_dir_final):
        os.makedirs(output_dir_final, exist_ok=True)
    
    trainer.train()
    trainer.save_model(output_dir_final)
    
    if ddp:
      dist.destroy_process_group()