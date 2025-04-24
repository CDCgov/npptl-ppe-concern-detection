# Randomly sample 3,200 PPE-related complaints for manual labeling

import pandas as pd
import numpy as np
import random

osha = pd.read_csv("../01_clean-data/osha_clean_05.csv", dtype={'Site.Zip': object})
osha_ppe = osha[osha['isPPE']]

# Randomly sample N records for the study sample
N = 3200

# Randomize order of PPE-related complaints
osha_ppe_randomized = osha_ppe.sample(frac=1)

# Randomly sample N PPE-related complaints for the study sample
osha_ppe_samp = osha_ppe_randomized[0:N]

# Unsampled PPE complaints
osha_ppe_unsamp = osha_ppe_randomized[N:]

# Keep only UPA.ID and Hazard.Desc.Loc (easier to read than lowercased) columns for ML dataset.
osha_ppe_samp_distinct = osha_ppe_samp.drop_duplicates(subset=['Hazard.Desc.Loc.lt'])[['UPA.ID', 'Hazard.Desc.Loc']]

# Write to file
osha_ppe_samp.to_csv('ppe_sample.csv', index = False)
osha_ppe_samp_distinct.to_csv('ppe_sample_distinct.csv', index = False)
osha_ppe_unsamp.to_csv('ppe_unsampled.csv', index = False)