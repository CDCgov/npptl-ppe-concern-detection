# Additional cleanup of data file

library(tidyverse)
library(lubridate)

osha = read.csv("osha_clean_04.csv")
osha = osha %>% rename("UPA.ID"="UPA..",
                        "Insp.Number"="Insp..",
                        "Hazard.Desc.Loc"="Hazard.Desc...Location",
                        "Hazard.Desc.Loc.lt"='Hazard.Desc...Location.lt',
                        "Alleged.Hazards.Emp.Exposed"="X..Alleged.Hazards..Emp.Exposed")
osha = osha %>% select(-c("X2022.NAICS.Title", 
                          "X2017.NAICS.Title", 
                          "X2012.NAICS.Title", 
                          "X2007.NAICS.Title", 
                          "X2002.NAICS.Title",
                          "X2022.NAICS.Title.3", 
                          "X2017.NAICS.Title.3", 
                          "X2012.NAICS.Title.3", 
                          "X2007.NAICS.Title.3",
                          "NAICS.Code.4"))
# Trim whitespace
tmp = c('Estab.Name', 
        'UPA.ID',
        'DBA',
        'Site.Address.1',
        'Site.Address.2',
        'Site.City',
        'Site.State',
        'Site.Zip',
        'Site.County',
        'Receipt.Type',
        'Formality',
        'Insp.Number',
        'Hazard.Desc.Loc',
        'Hazard.Desc.Loc.lt',
        'Alleged.Hazards.Emp.Exposed',
        'Language.Detected')
osha = osha %>% mutate_at(tmp, str_trim)

# Alleged.Hazards.Emp.Exposed
tmp = str_split(osha$Alleged.Hazards.Emp.Exposed, '/', ) %>% sapply(function(x) as.numeric(x)) %>% t
osha = osha %>% mutate(Num.Alleged.Hazards = tmp[,1])
osha = osha %>% mutate(Num.Emp.Exposed = tmp[,2])
                                                                    
# Fix ZIP codes
osha = osha %>% 
mutate(Site.Zip = na_if(Site.Zip, 'UNKWN')) %>%
mutate(Site.Zip = na_if(Site.Zip, '')) %>%
mutate(Site.Zip = sprintf("%05d", as.numeric(Site.Zip))) %>%
mutate(Site.Zip = as.character(Site.Zip))
                                                                    
# Site.City and Site.State
osha = osha %>%
mutate(Site.City = na_if(Site.City, 'Added for legacy migration')) %>%
mutate(Site.City = na_if(Site.City, '')) %>%
mutate(Site.State = na_if(Site.State, 'UK')) %>%
mutate(Site.State = na_if(Site.State, ''))

# Reorder cols & select relevant cols
osha = osha %>%
select(UPA.ID, UPA.Receipt.Date,
       Estab.Name, DBA,
       Site.Address.1, Site.Address.2, Site.City, Site.State, Site.Zip, Site.County,
       RID, Receipt.Type, Formality,
       Insp.Number, Num.Alleged.Hazards, Num.Emp.Exposed,
       Hazard.Desc.Loc, Hazard.Desc.Loc.lt,
       NAICS.Code, NAICS.Title.6, NAICS.Code.3, NAICS.Title.3, NAICS.Title.2, NAICS.Code.2,
       Language.Detected, isPPE,
       Primary.NAICS, Site.NAICS)

# Drop complaints which have no narrative
osha = osha %>% filter(Hazard.Desc.Loc!='')

# Write to file
write.csv(osha, "osha_clean_05.csv", row.names = FALSE) # 78710, 28