
# Enhancing Cyber Threat Intelligence with Transformer-based LLMs

## Overview

This repository contains the code and datasets for the undergraduate thesis project titled "Enhancing Cyber Threat Intelligence with Transformer-based LLMs." The goal of this project is to improve Named Entity Recognition (NER) for Cyber Threat Intelligence (CTI) using large language models (LLMs), enabling organizations to proactively defend against various cyber threats.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Introduction

In this project, we explore the application of transformer-based models like BERT and RoBERTa to enhance NER tasks within the domain of CTI. By leveraging the power of LLMs, we aim to outperform traditional NLP techniques in accurately identifying and classifying entities in cybersecurity-related text.

## Datasets

We used two widely recognized cybersecurity datasets for this project:
- DNRTI
- APTNER

The datasets were preprocessed and structured into CSV files for training and evaluation purposes.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/NafisMahi/NER-in-CTI-with-LLMs.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the datasets:
    ```python
    import datasets
    train_dataset = datasets.load_dataset('csv', data_files='path/to/train.csv')
    valid_dataset = datasets.load_dataset('csv', data_files='path/to/valid.csv')
    test_dataset = datasets.load_dataset('csv', data_files='path/to/test.csv')
    ```

2. Tokenize and align labels:
    ```python
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    ```

3. Fine-tune the transformer model:
    ```python
    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    trainer.train()
    ```

4. Evaluate the model:
    ```python
    metrics = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    print(metrics)
    ```

## Results

Our experiments demonstrated that the SecureBERT model fine-tuned on the DNRTI and APTNER datasets outperformed state-of-the-art metrics in NER for CTI, establishing that LLMs are superior to traditional NLP techniques in this domain.

## Contributors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
