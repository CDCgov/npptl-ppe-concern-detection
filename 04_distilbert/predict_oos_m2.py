import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import numpy as np
import os

# Create directories if they do not exist
output_dir = "predict_oos_m2/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ppe_unsampled = pd.read_csv("../02_sample-data/ppe_unsampled.csv")

model = AutoModelForSequenceClassification.from_pretrained("model/model2/", local_files_only=True, from_tf=False, config="model/model2/config.json")
tokenizer = AutoTokenizer.from_pretrained("model/model2/")
pipe = pipeline(task='text-classification', model=model, tokenizer=tokenizer, max_length=512, truncation=True)

def pred_probs_as_dict(text):
    x = pipe(text, return_all_scores=True)
    keys = [d['label'] for d in x[0]]
    values = [d['score'] for d in x[0]]
    dictionary = dict(zip(keys, values))
    return dictionary

probs_df = ppe_unsampled[['Hazard.Desc.Loc.lt']].apply(lambda text: pred_probs_as_dict(text.iloc[0]), axis='columns', result_type='expand')
ppe_unsampled = pd.concat([ppe_unsampled, probs_df], axis='columns')
ppe_unsampled.to_csv("{}ppe_unsampled_probs.csv".format(output_dir), index=False)


ppe_unsampled2 = pd.read_csv("../02_sample-data/ppe_unsampled.csv")
probs = pd.read_csv("{}ppe_unsampled_probs.csv".format(output_dir))
preds = probs[['Availability',
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
            'DisinfectionMaintenance']]
preds = preds >= 0.5
print(preds.sum())
preds_df = pd.concat([ppe_unsampled2.reset_index(drop=True), preds.reset_index(drop=True)], axis='columns')
preds_df.to_csv("{}ppe_unsampled_preds.csv".format(output_dir), index=False)