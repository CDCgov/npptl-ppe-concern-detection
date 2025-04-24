# Identify PPE-related complaints. This script creates a variable `isPPE` which flags complaints
# containing PPE-related keywords in ppe_terms.csv

import pandas as pd
from functools import reduce
import re

osha = pd.read_csv("osha_clean_03.csv", dtype={'Site.Zip': object})
ppe_terms = pd.read_csv("ppe_terms.csv")
ppe_terms = ppe_terms['PPE Term'].to_list()

# Handle regexes separately
ppe_terms = ['respirator\\W' if x==r'respirator\\W' else x for x in ppe_terms]
ppe_terms = ['respirator\\W' if x==r'respirator\\W' else x for x in ppe_terms]

osha['Hazard Desc & Location lt'] = osha['Hazard Desc & Location lt'].fillna('')
ppe_term_occurred = {}
for term in ppe_terms:
    print(term)
    found = osha['Hazard Desc & Location lt'].apply(lambda x: bool(re.search(term, x)))
    ppe_term_occurred[term] = found
    
ppe_occurred = reduce(lambda x, y : x + y, ppe_term_occurred.values())

osha_ppe = osha[ppe_occurred] # 30999, 37
osha_nonppe = osha[~ppe_occurred] # 47771, 37

# Proportion of complaints identified as PPE-related
osha_ppe.shape[0] / osha.shape[0] # 0.3935

# Create temporary Excel file where
# 1st sheet contains complaints ID'd as PPE-related
# 2nd sheet contains complaints ID'd as not PPE-related
# Examine both sheets to identify additional PPE words and refine filtering
writer = pd.ExcelWriter('ppe_vs_nonppe.xlsx', engine='openpyxl')
osha_ppe[['Hazard Desc & Location lt', 'Hazard Desc & Location']].to_excel(writer, sheet_name = 'PPE', index = False)
osha_nonppe[['Hazard Desc & Location lt', 'Hazard Desc & Location']].to_excel(writer, sheet_name = 'Non-PPE', index = False)
writer.close()

# Add a PPE flag (isPPE) to the OSHA complaint data
osha['isPPE'] = ppe_occurred

# Write to file
osha.to_csv('osha_clean_04.csv', index = False) # 78770, 38