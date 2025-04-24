import pandas as pd
import numpy as np
from functools import reduce
import json
import requests
import pickle
import numpy as np
import os
import glob

# Function from https://tech.marksblogg.com/language-translation-ai-python.html
def detect(text):
    resp = requests.post('http://127.0.0.1:5000/detect',
                         data={'q': text})
    return json.loads(resp.content)

if __name__ == "__main__":
    osha = pd.read_csv("osha_clean_01.csv")

    # Convert all NaN in Hazard Desc & Location to empty string ('') before lowercasing, etc.
    osha[['Hazard Desc & Location']] = osha[['Hazard Desc & Location']].fillna('')
    print("Number of empty Hazard Desc & Location: ")
    print(osha[osha['Hazard Desc & Location']==''].shape) # 60, 35

    # Lowercase
    osha['Hazard Desc & Location LOWERCASE'] = osha['Hazard Desc & Location'].apply(lambda x: x.lower())
    
    # Process in chunks
    output_dir = "language_detections/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_chunks = 32
    chunks = np.array_split(osha, n_chunks)
    for i in range(len(chunks)):
        print(f"Processing chunk {i}...")
        chunk = chunks[i]
        detection = chunk['Hazard Desc & Location LOWERCASE'].apply(lambda x: detect(x))
        with open(f'language_detections/lang_detected_{i}.pkl', 'wb') as file:
            pickle.dump(detection, file)
            
    # Add language detections
    list_of_files = glob.glob("language_detections/*.pkl")
    df = pd.concat(pd.read_pickle(f) for f in list_of_files)
    language_detected = df.apply(lambda x: x[0]['language'] if isinstance(x, list) and len(x) > 0 else None)
    print(language_detected.value_counts(dropna=False))
    osha['Language Detected'] = language_detected
    
    # Write to file
    osha.to_csv('./osha_clean_02.csv', index = False) # 78770, 36