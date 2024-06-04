<div align="center">
  <h3><b> STFEN: Spatial and Temporal Feature Extraction Networks for Air Pollution Modeling. </b></h3>
</div>

---



If you find this repository useful for your work, please consider citing it as [such](./citation.bib).


### Developing with STFEN

## 📦 Built-in Baselines

### Baselines

- Informer, PatchTST, DLinear, NLinear, STID, Crossformer, ...
- VGG, RestNet

## 💿 Dependencies

<details>
  <summary><b>Preliminaries</b></summary>


### OS

We recommend using STFEN on Linux systems (*e.g.* Ubuntu and CentOS). 
Other systems (*e.g.*, Windows and macOS) have not been tested.

### Python

Python >= 3.6 (recommended >= 3.9).

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended to create a virtual python environment.

### Other Dependencies
</details>

STFEN is built based on PyTorch and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**After ensuring** that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

### Warning

STFEN is built on PyTorch 1.9.1 or 1.10.0, while other versions have not been tested.


## 🎯 Getting Started of Developing with STFEN

### 1 Preparing Data

- **Clone STFEN**

    ```bash
    cd /path/to/your/project
    git clone https://github.com/wangs2020/STFEN
    ```

- **Pre-process Data**

    ```bash
    cd /path/to/your/project
    python scripts/data_preparation/pm1/generate_training_data.py
    ```

### 2 Steps to Evaluate the Model


- **Configure the Configuration File**

    You can configure all the details of the pipeline and hyperparameters in a configuration file, *i.e.*, **everything is based on config**.
    The configuration file is a `.py` file, in which you can import your model and runner and set all the options.
    An example of the configuration file can be found in [baselines/STFEN/pm1_stfen.py](baselines/STFEN/pm1_stfen.py)

### 3 Run It!

- **Reproducing Built-in Models**

  ```bash
  python experiments/train_stfen.py -c baselines/STFEN/pm1_stfen.py --gpus '0'
  ```

- **Customized Your Own Model**

  [Example](baselines/STFEN)


## 📉 Main Results

in preparation

## 🔗 Acknowledgement

STFEN is developed based on [BasicTS](https://github.com/zezhishao/BasicTS) and [EasyTorch](https://github.com/cnstark/easytorch)
