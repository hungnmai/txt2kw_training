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

Before training the model, make sure to configure the following parameters:


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
  "save_folder": "./models",
  "file_train": "./data/title.txt",
  "folder_save": "./data"
}
```

Here's a brief explanation of each parameter:

* epoch_num: The number of training epochs to perform. You should set 2-3 epochs to get good result.
* train_batch_size: The batch size for training data. Modify this value depending on your available resources.
* eval_batch_size: The batch size for evaluation data. Adjust as needed.
* save_folder: The folder where the trained model will be saved.
* file_train: The path to the text file containing the training data.
* file_train: Folder to save data training

Make sure to update these parameters according to your specific requirements before initiating the training process.

#### 2.3 Training model

```shell
conda activate keyword2title
python train_bart.py
```
