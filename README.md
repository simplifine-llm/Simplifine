# 🌟 Simplifine 🌟

## Super-Easy, Open-Source Cloud-Based LLM Finetuning

**Try here –**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simplifine-llm/Simplifine/blob/main/examples/cloud_quickstart.ipynb)

### **Get a FREE API Key for  FINETUNING [HERE](https://app.simplifine.com/#/signup)**


Simplifine streamlines LLM finetuning on any dataset or model with one simple command, handling all infrastructure, job management, cloud model storage, and inference.

## Features
- **🚀 Easy Cloud-Based LLM Finetuning:** Fine-tune any LLM with just one command.

- **☁️ Seamless Cloud Integration:** Automatically manage the downloading, storing, and running of models directly from the cloud.

- **🤖 Built-in AI Assistance:** Get help with hyperparameter selection, synthetic dataset generation, and data quality checks.

- **🔄 On-Device to Cloud Switching:** Add a simple decorator to transition from local to cloud-based training.

- **⚡ Auto-Optimization:** Automatically optimizes model and data parallelization Unsloth (*coming soon!*), Deepspeed ✅ and FDSP ✅

- **📊 Custom Evaluation Support:** Use the built-in LLM for evaluations functions or import your own custom evaluation metrics.

- **💼 Community Support:** Asking any support questions on the Simplifine Community Discord.

- **🏅 Trusted by Leading Institutions:** Research labs at the University of Oxford rely on Simplifine for their LLM finetuning needs.

---

## 🏁 Quickstart

Get started here > [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simplifine-llm/Simplifine/blob/main/examples/cloud_quickstart.ipynb)


## 📚 Documentation

Find our full documentation at [docs.simplifine.com](http://docs.simplifine.com).

## 📦 Installation

Installing from PyPI
```bash
pip install simplifine-alpha
```

You can also directly install from github using the following command:
```bash
pip install git+https://github.com/simplifine-llm/Simplifine.git
```

## 🤝 Contributing

We are looking for contributors! Please send an email to [founders@simplifine.com](mailto:founders@simplifine.com) to get onboarded! We welcome all types of contributions.

## 📄 License

Simplifine is licensed under the GNU General Public License Version 3. See the LICENSE file for more details.


## 💬 Support

If you have any suggestions for new features you'd like to see implemented, please raise an issue—we will work hard to make it happen ASAP! For any other questions, feel free to contact us at [founders@simplifine.com](mailto:founders@simplifine.com).



## 🔄 Updates

#### **v0.0.8**
- **🐛 Bug Fixes:** Streamlined code and resolved trainer-related issues for smoother operation.
- **✨ New Feature:** Introduced support for defining more complex configuration files, enhancing the flexibility of the trainer.
- **📚 Documentation:** Added new examples, including tutorials on cloud-based training and creating a fake news detector.
- **🔗 Updated Documentation:** Check out the latest docs at [docs.simplifine.com](https://docs.simplifine.com).

#### **v0.0.71**
- **🐛 Bug Fixes:** Fixed issues that caused loading failures on certain configurations, ensuring broader compatibility.
- **✨ New Feature:** Enabled direct installation from Git and added support for Hugging Face API Tokens, allowing access to restricted models.
- **📚 Documentation:** Refreshed examples to reflect the latest features.



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

## 🪲 FAQs

**Issue: RuntimeError: Error building extension 'cpu_adam' python dev**

This error occurs when `python-dev` is not installed, and ZeRO is using offload. To resolve this, try:

```bash
# Try sudo apt-get install python3-dev if the following fails.
apt-get install python-dev   # for Python 2.x installs
apt-get install python3-dev  # for Python 3.x installs
``` 

See this [link](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory)
