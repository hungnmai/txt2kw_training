import json
import torch
import logging
import os
import datetime
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from transformers.trainer import Trainer
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict
from kw_extract.kw_yake import gen_training_data
from transformers import BartForConditionalGeneration


# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_qa_data(file_train, temp_data_folder):
    train_data, dev_data = gen_training_data(file_train, temp_data_folder)
    print("train_data: ", len(train_data))
    print("train_data: ", len(dev_data))
    dataset = DatasetDict(
        {"train": Dataset.from_list(train_data),
         "dev": Dataset.from_list(dev_data)}
    )
    return dataset


def print_random_records(ds, tokenizer):
    # logging.info(f'padding_id: {tokenizer.pad_token_id}')
    data_loader = DataLoader(ds, batch_size=3, collate_fn=customize_collate)
    count = 0
    for batch in data_loader:
        if count > 5:
            continue
        # logging.info('---------------------------------')
        # logging.info(f"""length of this batch: {batch['input_ids'].size(-1)}""")
        # logging.info(batch)
        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            # logging.info('encoder: ', tokenizer.decode(batch['input_ids'][i].tolist()))
            # print('decoder: ', tokenizer.decode(batch['decoder_input_ids'][i].tolist()))
            label_ids = batch['labels'][i].tolist()
            label_ids = [item for item in label_ids if item != -100]
            # logging.info(f'labels: {tokenizer.decode(label_ids)}' )
        count += 1


def add_special_token(tokenizer):
    tokenizer.add_tokens(['<sep>'])


def pad_with_max_length(input_ids, max_leng, pad_id):
    cur_leng = len(input_ids)
    pad_leng = max_leng - cur_leng
    new_input_ids = input_ids + [pad_id for _ in range(pad_leng)]
    attention = [1 for _ in range(len(input_ids))] + [0 for _ in range(pad_leng)]
    return new_input_ids, attention


def get_encoded_kw(kw, b_tokenizer):
    result = []
    start, sep, end = b_tokenizer.encode('<sep>')
    for w in kw:
        wids = b_tokenizer.encode(' ' + w)[1:-1]
        result += wids + [sep]
    result = [start] + result[:-1] + [end]
    return result


def customize_collate(items):
    max_leng = max([len(item['input_ids']) for item in items])
    de_max_leng = max([len(item['decoder_input_ids']) for item in items])
    b_input_ids = []
    b_attention_mask = []
    b_decoded_ids = []
    b_labels = []
    b_decoder_attention_mask = []
    pad_id = 1
    for item in items:
        input_ids, attention_mask = pad_with_max_length(item['input_ids'], max_leng, pad_id)
        de_input_ids, de_attention_mask = pad_with_max_length(item['decoder_input_ids'], de_max_leng, pad_id)
        b_input_ids.append(input_ids)
        b_attention_mask.append(attention_mask)

        b_decoded_ids.append(de_input_ids)
        b_decoder_attention_mask.append(de_attention_mask)

        labels = item['decoder_input_ids'][1:] + [pad_id]
        labels, _ = pad_with_max_length(labels, de_max_leng, -100)
        b_labels.append(labels)
    batch = {
        'input_ids': torch.LongTensor(b_input_ids),
        'attention_mask': torch.LongTensor(b_attention_mask),
        # 'decoder_input_ids': torch.LongTensor(b_decoded_ids),
        'decoder_attention_mask': torch.LongTensor(b_decoder_attention_mask),
        'labels': torch.LongTensor(b_decoded_ids),
    }
    return batch


def get_kw_from_question(question):
    temp = question[7:].strip()
    parts = temp.split('<sep>')
    kw = [p.strip() for p in parts]
    return kw

def save_tokenizer_to_checkpoint(model_dir, tokenizer):
    for name in os.listdir(model_dir):
        if name.startswith("checkpoint-"):
            checkpoint_folder = os.path.join(model_dir, name)
            if os.path.isdir(checkpoint_folder):
                print("save tokenizer to: ", checkpoint_folder)
                tokenizer.save_pretrained(checkpoint_folder)

                
def train(config):
    logging.info("Start to training with config: ")
    logging.info(str(config))
    file_train = config['file_train']
    temp_data_folder = config['temp_data_folder']

    logging.info(f"File training: {file_train}")
    ds = read_qa_data(file_train, temp_data_folder)
    pretrained = config['pretrained']
    logging.info("Loading pretrained model: " + str(pretrained))
    tokenizer = BartTokenizer.from_pretrained(pretrained)
    add_special_token(tokenizer)
    tokenizer.save_pretrained(config['model_dir'])
    decoder_max_length = config['decoder_max_length']

    def preprocess_examples(batch):
        title_list = batch['title']
        spans_list = batch['spans']
        result = {'input_ids': [], 'decoder_input_ids': []}
        for i in range(len(title_list)):
            title = title_list[i]
            decoder_ids = tokenizer.encode(title, max_length=decoder_max_length, truncation=True)
            spans = spans_list[i]
            for kw in spans:
                input_ids = get_encoded_kw(kw, tokenizer)
                result['decoder_input_ids'].append(decoder_ids)
                result['input_ids'].append(input_ids)
        return result

    t1 = datetime.datetime.now()
    ds = ds.map(preprocess_examples, batched=True, remove_columns=['title', 'spans'])
    t2 = datetime.datetime.now()
    logging.info(f'Time for mapping data: {(t2 - t1)}')
    train_ds = ds['train']
    dev_ds = ds['dev']
    logging.info(f"Number of training dataset: {len(train_ds)}")
    logging.info(f"Number of valid dataset: {len(dev_ds)}")

    print_random_records(train_ds, tokenizer)
    # the next paragraph of code if about initializing models in jax/flax
    model = BartForConditionalGeneration.from_pretrained(pretrained)
    model.resize_token_embeddings(len(tokenizer))
    train_args = TrainingArguments(
        disable_tqdm=False,
        output_dir=config.get('model_dir', "models"),
        num_train_epochs=config.get("epoch_num", 3),  # total # of training epochs
        per_device_train_batch_size=config.get("train_batch_size", 64),  # batch size per device during training
        per_device_eval_batch_size=config.get('eval_batch_size', 64),  # batch size for evaluation
        warmup_steps=config.get('warm_up', 3),  # number of warmup steps for learning rate scheduler
        weight_decay=config.get('weight_decay', 0.01),  # strength of weight decay
        logging_dir=config.get('model_dir', "models"),  # directory for storing logs
        learning_rate=config.get("learning_rate", 4e-5),
        gradient_accumulation_steps=1,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=train_args,  # training arguments, defined above
        train_dataset=train_ds,  # training dataset
        eval_dataset=dev_ds,  # evaluation dataset
        data_collator=customize_collate,
        # compute_metrics=compute_metrics
    )
    t1 = datetime.datetime.now()
    if 'check_point' in config:
        path_to_checkpoint = config['model_dir'] + '/' + config['check_point']
        logging.info(f'Continue to train from checkpoint: {path_to_checkpoint}', )
        trainer.train(path_to_checkpoint)
    else:
        trainer.train()
    t2 = datetime.datetime.now()
    trainer.evaluate()
    save_tokenizer_to_checkpoint(config['model_dir'], tokenizer)
    logging.info(f'Training time: {(t2 - t1)} seconds')


def get_default_config(file_config) -> dict:
    """load configs in json file"""
    with open(file_config, "r") as fp:
        configs = json.load(fp)
        return configs


def main(file_config: str):
    config = get_default_config(file_config)
    train(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--config',
                        type=str,
                        help='file config',
                        required=True)

    args = parser.parse_args()
    print(args)
    main(args.config)
