# 🌟 Simplifine 🌟

**The easiest, fully open-source LLM finetuning library!**

Simplifine lets you invoke LLM finetuning with just one line of code using any Hugging Face dataset or model.

## 🚀 Features

- **Supervised Fine Tuning** 🧑‍🏫
- **Question-Answer Finetuning** ❓➕
- **Contrastive Loss for Embedding Tasks** 🌌
- **Multi-label Classification Finetuning** 🏷️
- **WandB Logging** 📊
- **In-built Evaluation Tools** 📈
- **Automated Finetuning Parameters** 🤖
- **State-of-the-art Optimization Techniques (DeepSpeed, FDSP)** 🏎️

## 📦 Installation

```bash
pip install simplifine-alpha
```

Or you can install the package from source. To do so, simply download the content of this repository and navigate to the installation folder and run the following command:

```bash
pip install .
```

You can also directly install from github using the following command:
```bash
pip install git+https://github.com/simplifine-llm/Simplifine.git
```

## 🏁 Quickstart
```python
from simplifine_alpha import train_engine
import time

alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

alpaca_keys = ['instruction', 'input', 'output']

print("Training model...")

train_engine.hf_sft(
    model_name='unsloth/llama-3-8b-bnb-4bit',
    dataset_name='yahma/alpaca-cleaned',
    keys=alpaca_keys,
    template=alpaca_template,
    num_epochs=1,
    batch_size=2,
    lr=2e-4,
    from_hf=True,
    response_template='### Response:',
    use_peft=False,
    peft_config=None,
    ddp=False,
    zero=True
)
```

## 🤝 Contributing

We are looking for contributors! Please send an email to [founders@simplifine.com](mailto:founders@simplifine.com) to get onboarded, or add your name to the waitlist on [www.simplifine.com](http://www.simplifine.com).

## 📄 License

Simplifine is licensed under the GNU General Public License Version 3. See the LICENSE file for more details.

## 📚 Documentation

Find our full documentation at [docs.simplifine.com](http://docs.simplifine.com).

## 💬 Support

Please raise issues for any new features you would like to see implemented—we will work hard to make it happen ASAP! For any other questions, contact us at [founders@simplifine.com](mailto:founders@simplifine.com).

