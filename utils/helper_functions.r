library(tidyverse)
library(lubridate)

# Loads and reformats osha_clean_04.csv for analysis in R
prep_osha_R = function(fpath_data) {
  osha = read.csv(fpath_data, colClasses = c(Site.Zip = "character")) # Keeps the leading 0 in the ZIP codes
  print(dim(osha))
  osha$Site.State = na_if(osha$Site.State, '')
  osha = mutate(osha, 
                UPA.Receipt.Date = date(as_datetime(UPA.Receipt.Date)),
                Site.City = na_if(Site.City, ''),
                Site.State = na_if(Site.State, ''),
                Site.Zip = na_if(Site.Zip, ''),
                Site.County = na_if(Site.County, ''),
                Site.Address.1 = na_if(Site.Address.1, ''),
                Site.Address.2 = na_if(Site.Address.2, ''),
                
                Hazard.Desc.Loc = na_if(Hazard.Desc.Loc, ''),
                Hazard.Desc.Loc.lt = na_if(Hazard.Desc.Loc.lt, ''),
                
                isPPE = as.logical(isPPE),
                NAICS.Title.2 = factor(NAICS.Title.2),
                NAICS.Title.3 = factor(NAICS.Title.3),
                NAICS.Title.6 = factor(NAICS.Title.6),
                Receipt.Type = factor(Receipt.Type),
                Formality = factor(Formality))
  return(osha)
}