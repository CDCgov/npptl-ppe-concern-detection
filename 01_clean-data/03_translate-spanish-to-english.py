# Translate Spanish complaints into English using libretranslate

import pandas as pd
import numpy as np
from functools import reduce
import json
import requests

# Function from https://tech.marksblogg.com/language-translation-ai-python.html
def translate(text, from_lang):
    resp = requests.post('http://127.0.0.1:5000/translate',
                         data={'q': text,
                               'source': from_lang,
                               'target': 'en'})
    return json.loads(resp.content)

if __name__ == "__main__":
    osha = pd.read_csv("osha_clean_02.csv")
    
    # Convert all NaN in Hazard Desc & Location to empty string ('') before lowercasing, etc.
    osha[['Hazard Desc & Location']] = osha[['Hazard Desc & Location']].fillna('')
    print("Number of empty Hazard Desc & Location: ")
    print(osha[osha['Hazard Desc & Location']==''].shape) # 60, 35

    # Lowercase
    osha['Hazard Desc & Location LOWERCASE'] = osha['Hazard Desc & Location'].apply(lambda x: x.lower())

    # Translate Spanish complaints in chunks
    osha_nonenglish = osha[osha['Language Detected']=='es'] # 1328, 36
    print("Translating x1...")
    x1 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][0:100].apply(lambda x, from_lang='es': translate(x, from_lang))
    x2 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][100:200].apply(lambda x, from_lang='es': translate(x, from_lang))
    x3 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][200:300].apply(lambda x, from_lang='es': translate(x, from_lang))
    print("Translating x4...")
    x4 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][300:400].apply(lambda x, from_lang='es': translate(x, from_lang))
    x5 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][400:500].apply(lambda x, from_lang='es': translate(x, from_lang))
    x6 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][500:600].apply(lambda x, from_lang='es': translate(x, from_lang))
    print("Translating x7...")
    x7 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][600:700].apply(lambda x, from_lang='es': translate(x, from_lang))
    x8 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][700:800].apply(lambda x, from_lang='es': translate(x, from_lang))
    x9 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][800:900].apply(lambda x, from_lang='es': translate(x, from_lang))
    print("Translating x10...")
    x10 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][900:1000].apply(lambda x, from_lang='es': translate(x, from_lang))
    x11 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][1000:1100].apply(lambda x, from_lang='es': translate(x, from_lang))
    x12 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][1100:1200].apply(lambda x, from_lang='es': translate(x, from_lang))
    print("Translating x13...")
    x13 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][1200:1300].apply(lambda x, from_lang='es': translate(x, from_lang))
    x14 = osha_nonenglish['Hazard Desc & Location LOWERCASE'][1300:].apply(lambda x, from_lang='es': translate(x, from_lang))

    x_list = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]
    x_all = pd.concat(x_list, ignore_index=True)
    x_all = x_all.apply(lambda x: x['translatedText'])
    
    # Replace Hazard Desc & Location LOWERCASE with lowercased, translated narrative
    osha_nonenglish.loc[:,'Hazard Desc & Location LOWERCASE'] = x_all.values
    
    # Merge with English language complaints
    osha_english = osha[osha['Language Detected']=='en']
    osha_nolang = osha[osha['Language Detected'].isna()]
    osha_tr = pd.concat([osha_english, osha_nonenglish, osha_nolang], ignore_index=True) # ?? 78770, 37
    
    # Lowercase 'Hazard Desc & Location LOWERCASE' again to ensure it's lowercase
    osha_tr['Hazard Desc & Location LOWERCASE'] = osha_tr['Hazard Desc & Location LOWERCASE'].apply(lambda x: x.lower())
    
    # Rename columns
    #    - Hazard Desc & Location: original
    #    - Hazard Desc & Location lt: [l]owercased and [t]ranslated (if applicable) narrative
    osha_tr = osha_tr.rename(columns={"Hazard Desc & Location LOWERCASE": "Hazard Desc & Location lt"})
    
    # Write cleaned data to file
    osha_tr.to_csv('./osha_clean_03.csv', index = False) # 78770, 37