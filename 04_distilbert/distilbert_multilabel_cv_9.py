#!/usr/bin/env python
# coding: utf-8
# Limit CPU usage — MUST be set before any imports
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from torch.nn import functional as F
import pickle

# For cert issues
os.environ['CURL_CA_BUNDLE'] = '/etc/pki/tls/cert.pem'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/pki/tls/cert.pem'

def print_experimental_settings(test_size, labels, model_dir, output_dir, \
learning_rate, per_device_train_batch_size, per_device_eval_batch_size, \
num_train_epochs, weight_decay):
    '''
    Prints experimental settings.
    '''
    print("======Experimental Settings=====")
    print("DATA")
    print("train_size:".ljust(20), 1-test_size)
    print("num_labels:".ljust(20), len(labels))
    print("labels:".ljust(20), labels)
    print("\n")
    
    print("MODEL")    
    print("learning_rate:".ljust(30), learning_rate)
    print("per_device_train_batch_size:".ljust(30), per_device_train_batch_size)
    print("per_device_eval_batch_size:".ljust(30), per_device_eval_batch_size)
    print("num_train_epochs:".ljust(30), num_train_epochs)
    print("weight_decay:".ljust(30), weight_decay)
    print("model_dir:".ljust(30), model_dir)
    print("output_dir:".ljust(30), output_dir)

def compute_marginal_metrics(eval_pred, labels):
    '''
    Returns a dataframe of classification metrics computed marginally along the labels.
    '''
    y_true = pd.DataFrame(eval_pred.label_ids, columns=labels)
    y_pred, y_prob = get_predictions(eval_pred, labels)
    marginal_metrics = {}
    for label in labels:
        report = classification_report(y_true[label], y_pred[label], output_dict=True)
        macro_avg = report['macro avg']
        macro_avg['support_1_tst'] = report['1.0']['support']
        macro_avg['support_1_trn'] = df_train[label].sum()

        cm = confusion_matrix(y_true[label], y_pred[label]).flatten()
        macro_avg['true0_pred0'] = cm[0]
        macro_avg['true0_pred1'] = cm[1]
        macro_avg['true1_pred0'] = cm[2]
        macro_avg['true1_pred1'] = cm[3]

        marginal_metrics[label] = macro_avg
    marginal_metrics = pd.DataFrame(marginal_metrics).T
    ret = {'y_true':y_true, 'y_pred':y_pred, 'y_prob':y_prob, 'marginal_metrics':marginal_metrics}
    return(ret)
    
def multi_label_metrics(predictions, labels, threshold=0.5):
    '''
    Compute classification metrics for multilabel prediction.
    '''
    # Multilabel requires sigmoid - apply to predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # Finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    roc_auc_macro = roc_auc_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    # Return as dictionary
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'roc_auc_micro': roc_auc_micro,
               'roc_auc_macro': roc_auc_macro,
               'accuracy': accuracy}
    return metrics
    
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result
    
def preprocess_data(examples):
    text = examples[text_column_name]
    # encoding = tokenizer(text, padding="max_length", truncation=True)
    # We'll pad dynamically with the DataCollator
    encoding = tokenizer(text, truncation=True)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

def get_predictions(eval_pred, labels):
    '''
    Go from logit scores to 0/1 predictions.
    Args:
        - eval_pred: results of trainer.predict(). It has .metrics, .label_ids, .predictions (the logits).
    Values:
        - dataframe of 0s and 1s (same size as eval_pred.predictions), with labels as colnames.
        
    Requires: from torch.nn import functional as F
    '''
    logits = torch.from_numpy(eval_pred.predictions)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    y_pred = pd.DataFrame(predictions, columns=labels)
    y_prob = pd.DataFrame(probs, columns=labels)
    return y_pred, y_prob

def label_indicators_to_string(indicators, labels):
    '''
    Turns a 1D array of indicators into a string listing which labels are True.
    [1., 1., 0.], labels=['A', 'B', 'C'] --> "['A', 'B']"
    '''
    indicators = np.array(indicators, dtype=bool)
    true_labels = np.array(labels)[indicators]
    true_labels_string = str(true_labels.tolist())
    return true_labels_string

if __name__ == "__main__":
    # Params
    data_path = "../03_ppe-coding/ml_dataset.csv"
    eval_dir = "./evaluation/"
    output_dir = "./checkpoints/"
    model_dir = './model/'
    test_size = 0.25
    model_name = "distilbert-base-uncased"
    learning_rate=2e-5
    per_device_train_batch_size=16
    per_device_eval_batch_size=16
    num_train_epochs=20
    weight_decay=0.01
    
    # Create directories if they do not exist
    for directory in [eval_dir, output_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Prepare data
    text_column_name = 'Hazard.Desc.Loc.lt'
    labels = ['Availability', 
            'EnforceUse', 
            'NotWornE', 
            'WornIncorrectlyE', 
            'NotWornNE', 
            'NotWornU', 
            'EnforceCorrectUse', 
            'CrossContamination', 
            'PPEDiscouragedProhibited', 
            'Training', 
            'FitTest', 
            'Physiological', 
            'DisinfectionMaintenance'] # hard-coded
    df = pd.read_csv(data_path, encoding='utf-8')
    
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    # CV
    nsim = 15
    overall_metrics_list = []
    
    chunk = 9
    start = chunk*nsim
    for i_split in range(start, start+nsim):
        print("Split #{}".format(i_split))
          
        # Train-test split
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=i_split)
        df_train0 = df_train[[text_column_name] + labels].reset_index(drop=True)
        df_test0 = df_test[[text_column_name] + labels].reset_index(drop=True)
        
        # Convert to Hugging Face dataset
        train_dataset = Dataset.from_pandas(df_train0)
        test_dataset = Dataset.from_pandas(df_test0)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoded_train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
        encoded_test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names)

        encoded_train_dataset.set_format("torch")
        encoded_test_dataset.set_format("torch")

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained( \
            model_name,
            num_labels=len(labels),
            problem_type = "multi_label_classification",
            id2label=id2label,
            label2id=label2id)

        # Train model
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            logging_strategy="epoch"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset, # change for CV
            eval_dataset=encoded_test_dataset, # change to encoded_val_dataset for CV
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print_experimental_settings(test_size, labels, model_dir, output_dir, \
        learning_rate, per_device_train_batch_size, per_device_eval_batch_size, \
        num_train_epochs, weight_decay)

        trainer.train()
        
        ## Evaluate model
        ## Write to file
        model.eval() # put in testing mode - deactivates dropout modules
        eval_pred = trainer.predict(encoded_test_dataset)
        eval_pred_fname = "{}eval_pred_split{}.pkl".format(eval_dir, i_split)
        with open(eval_pred_fname, 'wb') as pickle_file:
            pickle.dump(eval_pred._asdict(), pickle_file)
        
        ## Overall metrics - add ith iteration
        overall_metrics_list.append(eval_pred.metrics)

        ## Marginal metrics
        ## Write to file
        mm = compute_marginal_metrics(eval_pred, labels)
        y_pred = mm['y_pred']
        y_prob = mm['y_prob']
        marginal_metrics = mm['marginal_metrics']
        marginal_metrics.to_csv("{}marginal_metrics_split{}.csv".format(eval_dir, i_split), encoding='utf-8')
        
        ## df_test: augment to allow for examination of predictions (true_labels, pred_labels, and pred_[label] indicators)
        ## Write to file - csv
        df_test['true_labels'] = df_test.apply(lambda x: label_indicators_to_string(x[labels], labels), axis=1)
        df_test['pred_labels'] = y_pred.apply(lambda x: label_indicators_to_string(x[labels], labels), axis=1)
        y_pred.columns = ['pred_'+label for label in labels]
        df_test = df_test.reset_index().join(y_pred)
        y_prob.columns = ['prob_'+label for label in labels]
        df_test = df_test.reset_index().join(y_prob)
        df_test.to_csv("{}df_test_split{}.csv".format(eval_dir, i_split), encoding='utf-8')
        
        # Save model
        model_subdir = model_dir + "model{}/".format(i_split)
        if not os.path.exists(model_subdir):
            os.makedirs(model_subdir)
        trainer.save_model(model_subdir)
        
        # Save eval dataset
        # encoded_test_dataset.save_to_disk(model_dir)
    
    ## Overall metrics - format
    ## Write to file
    overall_metrics = pd.DataFrame.from_dict(overall_metrics_list).T
    overall_metrics.to_csv("{}overall_metrics_{}.csv".format(eval_dir, chunk), encoding='utf-8')
