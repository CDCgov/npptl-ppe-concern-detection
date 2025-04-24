# Merge with NAICS datasets to get NAICS codes and titles

import pandas as pd
import numpy as np
from itertools import compress
from functools import reduce
import time
import random

## Collect all complaints through 07/15/2022
osha1 = pd.read_excel("../00_data-raw/OSHA/Closed_Federal_State_Plan_Valid_COVID-19_Complaints_Through_0429_2022.xlsx")
osha2 = pd.read_excel("../00_data-raw/OSHA/Closed_Federal_State_Plan_Valid_COVID-19_New_Complaints_0430_through_0715_2022.xlsx")
osha = pd.concat([osha1, osha2], ignore_index=True, sort=False)

## Prepare OSHA data for merging with NAICS data
osha['Primary NAICS'] = osha['Primary/Site NAICS'].transform(lambda x: x.split(' / ')[0])
osha['Site NAICS'] = osha['Primary/Site NAICS'].transform(lambda x: x.split(' / ')[1])

# Convert NAICS variables to integer type
osha['Primary NAICS'] = osha['Primary NAICS'].astype('int')

# One record has no Site NAICS, set it to '999999' before converting to int
osha.loc[osha['Site NAICS']=='', ['Site NAICS']] = '999999'
osha['Site NAICS'] = osha['Site NAICS'].astype('int')

## Prepare NAICS codes for merging
# Use Site NAICS when available. Otherwise, when Site NAICS is '' or 999999, use Primary NAICS.
# Less than 1% of complaints are missing Primary NAICS and Site NAICS.
# In approximately 5% of records [3685/(3685+75085)], the Primary and Site NAICS disagree.
osha['NAICS Code'] = osha['Site NAICS']
osha.loc[(osha['NAICS Code']==999999) & (osha['Primary NAICS']!=999999), ['NAICS Code']] = \
osha.loc[(osha['NAICS Code']==999999) & (osha['Primary NAICS']!=999999),:]['Primary NAICS']

## Prepare NAICS datasets for merging
naics2022 = pd.read_excel("../00_data-raw/NAICS/6-digit_2022_Codes.xlsx")
naics2017 = pd.read_excel("../00_data-raw/NAICS/6-digit_2017_Codes.xlsx")
naics2012 = pd.read_excel("../00_data-raw/NAICS/6-digit_2012_Codes.xlsx")
naics2007 = pd.read_excel("../00_data-raw/NAICS/6-digit_2007_Codes.xlsx")

naics2022.drop(0, inplace = True)
naics2017.drop(0, inplace = True)
naics2012.drop(0, inplace = True)
naics2007.drop(0, inplace = True)

naics2022.drop(['Unnamed: 2'], axis=1, inplace = True)
naics2017.drop(['Unnamed: 2'], axis=1, inplace = True)

# Rename to have common key
naics2022.rename(columns={'2022 NAICS Code':'NAICS Code'}, inplace=True)
naics2017.rename(columns={'2017 NAICS Code':'NAICS Code'}, inplace=True)
naics2012.rename(columns={'2012 NAICS Code':'NAICS Code'}, inplace=True)
naics2007.rename(columns={'2007 NAICS Code':'NAICS Code'}, inplace=True)

# Handle 2002 NAICS separately
naics2002 = pd.read_table("../00_data-raw/NAICS/6-digit_2002_Codes.txt", header=3, sep="  ")
naics2002.drop(columns=['2002 NAICS Title'], inplace=True)
naics2002.rename(columns={'Unnamed: 1':'2002 NAICS Title', 'Code':'NAICS Code'}, inplace=True)
naics2002.columns

naics2022['NAICS Code'] = naics2022['NAICS Code'].astype('int')
naics2017['NAICS Code'] = naics2017['NAICS Code'].astype('int')
naics2012['NAICS Code'] = naics2012['NAICS Code'].astype('int')
naics2007['NAICS Code'] = naics2007['NAICS Code'].astype('int')
naics2002['NAICS Code'] = naics2002['NAICS Code'].astype('int')

## Sequentially merge NAICS and OSHA data to get NAICS titles
# If no title found in 2022 NAICS, look in 2017 NAICS, etc. 

merged_osha = osha.merge(naics2022, on='NAICS Code', how='left')
merged_osha = merged_osha.merge(naics2017, on='NAICS Code', how='left')
merged_osha = merged_osha.merge(naics2012, on='NAICS Code', how='left')
merged_osha = merged_osha.merge(naics2007, on='NAICS Code', how='left')
merged_osha = merged_osha.merge(naics2002, on='NAICS Code', how='left')

def firstNonnull(d):
    a = list(d.values())
    if sum(pd.notna(a))==0:
        return(None)
    not_na = tuple(map(pd.notna, a))
    return(list(compress(a, not_na))[0])

# Sequential merge
x = merged_osha[['2022 NAICS Title', '2017 NAICS Title', '2012 NAICS Title', '2007 NAICS Title', '2002 NAICS Title']].to_dict(orient='records')
merged_osha['NAICS Title 6'] = list(map(firstNonnull, x))

# For NAICS codes without titles, research and add the title manually

# View NAICS codes without titles
# merged_osha[merged_osha['NAICS Title 6'].isna()]['NAICS Code'].value_counts()

# 722210 - Limited-Service Eating Places (CAN) in the 2002 NAICS. - https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=21823&CVD=21828&CPV=722210&CST=01012002&CLV=5&MLV=5
merged_osha.loc[(merged_osha['NAICS Code'] == 722210), 'NAICS Title 6'] = 'Limited-Service Eating Places'

# 56142 - Telephone Call Centers - https://www.census.gov/naics/?input=56142&year=2022&details=56142
merged_osha.loc[(merged_osha['NAICS Code'] == 56142), 'NAICS Title 6'] = 'Telephone Call Centers'

# 62211 - General Medical and Surgical Hospitals - https://www.census.gov/naics/?input=62211+&year=2022&details=62211
merged_osha.loc[(merged_osha['NAICS Code'] == 62211), 'NAICS Title 6'] = 'General Medical and Surgical Hospitals'

# 32541 - Pharmaceutical and Medicine Manufacturing - https://www.census.gov/naics/?input=32541&year=2022
merged_osha.loc[(merged_osha['NAICS Code'] == 32541), 'NAICS Title 6'] = 'Pharmaceutical and Medicine Manufacturing'

# Others
merged_osha.loc[(merged_osha['NAICS Code'] == 9211), 'NAICS Title 6'] = 'Executive, Legislative, and Other General Government Support'
merged_osha.loc[(merged_osha['NAICS Code'] == 56162), 'NAICS Title 6'] = 'Security Systems Services'
merged_osha.loc[(merged_osha['NAICS Code'] == 49319), 'NAICS Title 6'] = 'Other Warehousing and Storage'
merged_osha.loc[(merged_osha['NAICS Code'] == 72112), 'NAICS Title 6'] = 'Casino Hotels'
merged_osha.loc[(merged_osha['NAICS Code'] == 72221), 'NAICS Title 6'] = 'Limited-Service Eating Places'

### Add NAICS 3-digit codes & titles

# 999999 -> 999 but still acts as a marker for missingness
merged_osha['NAICS Code 3'] = merged_osha['NAICS Code'].apply(lambda x: int(str(x)[:3]))
merged_osha['NAICS Code 4'] = merged_osha['NAICS Code'].apply(lambda x: int(str(x)[:4]))

# 2022
naics2022_all = pd.read_excel('../00_data-raw/NAICS/2-6 digit_2022_Codes.xlsx')
naics2022_all = naics2022_all.iloc[1:]
ix_3digit = (naics2022_all.iloc[:,1].astype(str).str.len() == 3)
naics2022_3digit = naics2022_all.loc[ix_3digit]
naics2022_3digit = naics2022_3digit.rename(columns = {naics2022_3digit.columns[1]:'NAICS Code 3', 
                                                      naics2022_3digit.columns[2]:'2022 NAICS Title 3'})
naics2022_3digit = naics2022_3digit[['NAICS Code 3', '2022 NAICS Title 3']]
merged_osha2 = merged_osha.merge(naics2022_3digit, on='NAICS Code 3', how='left')

# 2017
naics2017_all = pd.read_excel('../00_data-raw/NAICS/2-6 digit_2017_Codes.xlsx')
naics2017_all = naics2017_all.iloc[1:]
ix_3digit = (naics2017_all.iloc[:,1].astype(str).str.len() == 3)
naics2017_3digit = naics2017_all.loc[ix_3digit]
ix_4digit = (naics2017_all.iloc[:,1].astype(str).str.len() == 4)
naics2017_4digit = naics2017_all.loc[ix_4digit]

naics2017_3digit = naics2017_3digit.rename(columns = {naics2017_3digit.columns[1]:'NAICS Code 3', 
                                                      naics2017_3digit.columns[2]:'2017 NAICS Title 3'})
naics2017_3digit = naics2017_3digit[['NAICS Code 3', '2017 NAICS Title 3']]
merged_osha2 = merged_osha2.merge(naics2017_3digit, on='NAICS Code 3', how='left')

# 2012
naics2012_all = pd.read_excel('../00_data-raw/NAICS/2-6 digit_2012_Codes.xls')
naics2012_all = naics2012_all.iloc[1:]
ix_3digit = (naics2012_all.iloc[:,1].astype(str).str.len() == 3)
naics2012_3digit = naics2012_all.loc[ix_3digit]
naics2012_3digit = naics2012_3digit.rename(columns = {naics2012_3digit.columns[1]:'NAICS Code 3', 
                                                      naics2012_3digit.columns[2]:'2012 NAICS Title 3'})
naics2012_3digit = naics2012_3digit[['NAICS Code 3', '2012 NAICS Title 3']]
merged_osha2 = merged_osha2.merge(naics2012_3digit, on='NAICS Code 3', how='left')

# 2007
naics2007_all = pd.read_excel('../00_data-raw/NAICS/2-6 digit_2007_Codes.xlsx')
naics2007_all = naics2007_all.iloc[1:]
ix_3digit = (naics2007_all.iloc[:,1].astype(str).str.len() == 3)
naics2007_3digit = naics2007_all.loc[ix_3digit]
naics2007_3digit = naics2007_3digit.rename(columns = {naics2007_3digit.columns[1]:'NAICS Code 3', 
                                                      naics2007_3digit.columns[2]:'2007 NAICS Title 3'})
naics2007_3digit = naics2007_3digit[['NAICS Code 3', '2007 NAICS Title 3']]
naics2007_3digit['NAICS Code 3'] = naics2007_3digit['NAICS Code 3'].astype(int)
merged_osha2 = merged_osha2.merge(naics2007_3digit, on='NAICS Code 3', how='left')

# All remaining records with no 3-digit title also have no 6-digit title.

# Sequential merge
x = merged_osha2[['2022 NAICS Title 3', '2017 NAICS Title 3', '2012 NAICS Title 3']].to_dict(orient='records')
merged_osha2['NAICS Title 3'] = list(map(firstNonnull, x))

## Add NAICS 2-digit codes & titles

# 999999 -> 99 but still acts as a marker for missingness
merged_osha2['NAICS Code 2'] = merged_osha2['NAICS Code'].apply(lambda x: int(str(x)[:2]))

# 2022
naics2022_all = pd.read_excel('../00_data-raw/NAICS/2-6 digit_2022_Codes.xlsx')
naics2022_all = naics2022_all.iloc[1:]
ix_2digit = (naics2022_all['2022 NAICS US   Code'].astype(str).str.len() == 2)
naics2022_2digit = naics2022_all.loc[ix_2digit]

naics2022_2digit = naics2022_2digit.rename(columns = {naics2022_2digit.columns[1]:'NAICS Code 2', 
                                                      naics2022_2digit.columns[2]:'2022 NAICS Title 2'})

# Add the following NAICS codes/titles from 2022 2-6 digit NAICS file to naics2022_2digit
# 31-33 Manufacturing
# 44-45 Retail Trade
# 48-49 Transportation and Warehousing
addtl = pd.DataFrame({'NAICS Code 2':list(range(31, 34)) + list(range(44, 46)) + list(range(48, 50)),
                    '2022 NAICS Title 2': ['Manufacturing']*3 + ['Retail Trade']*2 + ['Transportation and Warehousing']*2})
# naics2022_2digit = naics2022_2digit[['NAICS Code 2', '2022 NAICS Title 2']]append(addtl)
naics2022_2digit = pd.concat([naics2022_2digit[['NAICS Code 2', '2022 NAICS Title 2']], addtl], ignore_index=True)

merged_osha3 = merged_osha2.merge(naics2022_2digit, on='NAICS Code 2', how='left')

# All 2 digit codes were able to be mapped with 2022 NAICS data, so we didn't need to lookup 2-digit codes in prior years
# Therefore, rename 2022 NAICS Title 2 as NAICS Title 2
merged_osha3.rename(columns={'2022 NAICS Title 2':'NAICS Title 2'}, inplace=True)

## Translate complaints from Spanish -> English
osha = merged_osha3

## Strip leading and trailing whitespace in NAICS Titles
osha['NAICS Title 2'] = osha['NAICS Title 2'].str.strip()
osha['NAICS Title 3'] = osha['NAICS Title 3'].str.strip()
osha['NAICS Title 6'] = osha['NAICS Title 6'].str.strip()

## Write to file
osha.to_csv('./osha_clean_01.csv', index = False) # 78770, 35