# 🌟 Simplifine 🌟

**The easiest, fully open-source LLM finetuning library!**

Simplifine lets you invoke LLM finetuning with just one line of code using any Hugging Face dataset or model.

## Updates
## 🔄 Updates

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

Please raise issues for any new features you would like to see implemented—we will work hard to make it happen ASAP! For any other questions, contact us at [founders@simplifine.com](mailto:founders@simplifine.com).

## 🪲 FAQs and bugs
**RuntimeError: Error building extension 'cpu_adam' python dev**: This happens when python-dev is not installed and offload is being used by ZeRO. simple try 
```python
sudo apt-get install python-dev   # for python2.x installs
sudo apt-get install python3-dev  # for python3.x installs
```
See this [link](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory)
