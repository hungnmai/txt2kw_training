TRAINING MODEL KEYWORD TO TITLE

---

### 1. Installation and Setup


* Create a virtual environment using conda:
```shell
conda create -n keyword2title python=3.8
```

* Activate the environment:
```shell
conda activate keyword2title
```

* Install the required libraries:
```shell
cd txt2kw_training
pip install -r requirements.txt
```

### 2. Training the Model

#### 2.1 Training data
Before starting the training process, you need to prepare a dataset of titles. Each title should be on a separate line and saved in a text file. For example, you can use [title_example.txt](title_example.txt) as your dataset.

#### 2.2 Configuration

Before training the model, you need to create a json file with the same format like config.json, this will include required information for training:


```shell
{
  "pretrained": "facebook/bart-base",
  "learning_rate": 4e-5,
  "epoch_num": 3,
  "train_batch_size": 96,
  "eval_batch_size": 96,
  "weight_decay": 0.01,
  "seed": 10,
  "decoder_max_length": 80,
  "model_dir": "./models",
  "file_train": "./title_example.txt",
  "temp_data_folder": "./data"
}
```

Here's a brief explanation of each parameter:

* epoch_num: The number of training epochs to perform. You should set 2-3 epochs to get good result.
* train_batch_size: The batch size for training data. Modify this value depending on your available resources.
* eval_batch_size: The batch size for evaluation data. Adjust as needed.
* model_dir: The folder where the trained model will be saved.
* file_train: The path to the text file containing the training data.
* temp_data_folder: Folder to save generated training data.

Currently, train_batch_size is set as 96 to fit GPU: 16GB Tesla T4 (free colab GPU), If you have bigger GPU, you can try bigger value to reduce the training time. Basically, you should only change the <b>file_train, model_dir and temp_data_folder</b> if you don't understand other parameters.
This created json file will be the parameter for ``python train_bart.py``

#### 2.3 Training model

```shell
conda activate keyword2title
python train_bart.py --config path_to_config_file
```
path_to_config_file: is described in <b>2.2 Configuration</b>

For example:
```shell
python train_bart.py --config config.json
```
