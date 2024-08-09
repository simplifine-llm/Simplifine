# 🌟 Simplifine 🌟

**The easiest, fully open-source LLM finetuning library!**

Simplifine lets you invoke LLM finetuning with just one line of code using any Hugging Face dataset or model.

## Updates
## 🔄 Updates
**v0.0.8 (2024-08-08)**
- **Bug Fixes:** Code clean up and trainer fixes.
- **New Feature:** Ability to define more complex configuration files for the trainer.
- **Documentation:** -New examples on training cloud and training a fake news detector.
- **COMPREHENSIVE UPDATE of DOCUMENTATIONS on [docs.simplifine.com](https://docs.simplifine.com).**

**v0.0.71 (2024-07-25)**
- **Bug Fixes:** Resolved issues that prevented the library from loading on certain configurations.
- **New Feature:** Added support for installing directly from git. Added support for Hugging Face API Tokens to access restricted models.
- **Documentation:** Updated examples.

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

For a more comprehensive example, see this [notebook](https://github.com/simplifine-llm/Simplifine/blob/main/examples/cloud_quickstart.ipynb) in the examples folder:

Further examples on how to use train engine are also located in the examples folder.

## 🤝 Contributing

We are looking for contributors! Please send an email to [founders@simplifine.com](mailto:founders@simplifine.com) to get onboarded, or add your name to the waitlist on [www.simplifine.com](http://www.simplifine.com).

## 📄 License

Simplifine is licensed under the GNU General Public License Version 3. See the LICENSE file for more details.

## 📚 Documentation

Find our full documentation at [docs.simplifine.com](http://docs.simplifine.com).

## 💬 Support

If you have any suggestions for new features you'd like to see implemented, please raise an issue—we will work hard to make it happen ASAP! For any other questions, feel free to contact us at [founders@simplifine.com](mailto:founders@simplifine.com).

## ⛮ General Compute Considerations

We currently support both DistributedDataParallel (DDP) and ZeRO from DeepSpeed.

**TL;DR**: 
- **DDP** is useful when a model can fit in GPU memory (this includes gradients and activation states).
- **ZeRO** is useful when a model requires sharding across multiple GPUs.

**Longer Version**:

- **DDP**: Distributed Data Parallel (DDP) creates a replica of the model on each processor (GPU). For example, imagine 8 GPUs, each being fed a single data point—this would make a batch size of 8. The model replicas are then updated on each device. DDP speeds up training by parallelizing the data-feeding process. However, DDP **fails** if the replica cannot fit in GPU memory. Remember, the memory not only hosts parameters but also gradients and optimizer states.

- **ZeRO**: ZeRO is a powerful optimization developed by DeepSpeed and comes in different stages (1, 2, and 3). Each stage shards different parts of the training process (parameters, gradients, and activation states). This is really useful if a model cannot fit in GPU memory. ZeRO also supports offloading to the CPU, making even more room for training larger models.

### Example Scenarios and Appropriate Optimization Methods:
1. **LLaMA-3-8b model with 16-bit precision**: Use ZeRO Stage 3 on 8 A100s.
2. **LLaMA-3-8b model with LoRA adapters**: Usually fine with DDP on A100s.
3. **GPT-2 with 16-bit precision**: Use DDP.

## 🪲 FAQs and Bugs

**Issue: RuntimeError: Error building extension 'cpu_adam' python dev**

This error occurs when `python-dev` is not installed, and ZeRO is using offload. To resolve this, try:

```bash
# Try sudo apt-get install python3-dev if the following fails.
apt-get install python-dev   # for Python 2.x installs
apt-get install python3-dev  # for Python 3.x installs
``` 

See this [link](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory)
