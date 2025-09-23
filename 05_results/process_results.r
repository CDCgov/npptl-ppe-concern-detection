# Process results
# Produces three CSV files:
# -- results_marginal.csv (marginalresults for simulations)
# -- results_overall.csv (overall results for simulations)
# -- ppe_labeled.csv (predictions and manually labeled/inferred labels for all PPE-related complaints)

library(tidyr)
library(plyr)
library(dplyr)
library(ggplot2)
source("../utils/helper_functions.r")

# -------------------------------------------------------
# results_marginal.csv
# -------------------------------------------------------

sims = list.files(path = "../04_distilbert/evaluation", pattern="^marginal_metrics", full.names = TRUE)

dfs = lapply(sims, function(fpath) {
    x = read.csv(fpath) %>% rename(., concern=X)
    x$sim = as.numeric(str_match(fpath, "split_(.*).csv")[,2])
    x = relocate(x, sim)
    return(x)
})

df = bind_rows(dfs)
print(dim(df))
print(length(sims))

df = df %>% mutate(accuracy = (true0_pred0 + true1_pred1) / (true0_pred0 + true1_pred1 + true0_pred1 + true1_pred0)) # per-label accuracies

write.csv(df, "./results_marginal.csv", row.names=FALSE)

# -------------------------------------------------------
# results_overall.csv
# -------------------------------------------------------

sims = list.files(path = "../04_distilbert/evaluation", pattern="^overall_metrics", full.names = TRUE)

dfs = lapply(sims, function(fpath) {
    x = read.csv(fpath) %>% pivot_wider(names_from=X, values_from=X0)
    x$sim = as.numeric(str_match(fpath, "overall_metrics_(.*).csv")[,2])
    x = relocate(x, sim)
    return(x)
})

df = bind_rows(dfs)
print(dim(df))
print(length(sims))

write.csv(df, "./results_overall.csv", row.names=FALSE)

# -------------------------------------------------------
# ppe_labeled.csv
# -------------------------------------------------------

# Majority vote predictions on OOS complaints
# - Merges df with sample info (NAICS, etc.)
# - Includes all complaints - study sample AND oos.
# - Creates a new variable that can be used to filter for study sample OR oos as desired.
# - Writes csv file

labels = c('Availability', 'EnforceUse', 'NotWornE', 'WornIncorrectlyE', 
           'NotWornNE', 'NotWornU', 'EnforceCorrectUse', 'CrossContamination',
           'PPEDiscouragedProhibited', 'Training', 'FitTest', 'Physiological',
           'DisinfectionMaintenance')

p0 = prep_osha_R("../04_distilbert/predict_oos_m0/ppe_unsampled_preds.csv")
p1 = prep_osha_R("../04_distilbert/predict_oos_m1/ppe_unsampled_preds.csv")
p2 = prep_osha_R("../04_distilbert/predict_oos_m2/ppe_unsampled_preds.csv")

p0 = p0 %>% mutate_at(labels, as.logical)
p1 = p1 %>% mutate_at(labels, as.logical)
p2 = p2 %>% mutate_at(labels, as.logical)

rowwise_majority_vote = function(x) {
    y = apply(x, 1, function(v) {
        majority_vote = names(sort(table(v), decreasing=TRUE))[1]
        majority_vote = as.logical(majority_vote)
        return(majority_vote)
    })
    return(y)
}

votes = sapply(labels, function(label) {
    x0 = p0[,label]
    x1 = p1[,label]
    x2 = p2[,label]
    x = bind_cols(pred0=x0, pred1=x1, pred2=x2)
    y = rowwise_majority_vote(x)
    return(y)
})

votes = as.data.frame(votes)

# p0 will contain OOS predictions for OOS complaints
p0[,labels] = votes

# Map labels for distinct narratives back to study sample
ppe_sample = prep_osha_R("../02_sample-data/ppe_sample.csv")
ml = read.csv("../03_ppe-coding/ml_dataset.csv") %>% select(all_of(c('Hazard.Desc.Loc.lt', labels)))
study_sample = left_join(ppe_sample, ml, by="Hazard.Desc.Loc.lt")

binarize = function(x) (ifelse(x=='', 0, 1))
study_sample = study_sample %>% mutate_at(labels, binarize)

# Combine manually labeled sampled complaints + ML labeled OOS complaints
keep_cols = intersect(colnames(study_sample), colnames(p0))
ppe_labeled = bind_rows(
    study_sample %>% select(all_of(keep_cols)),
    p0 %>% select(all_of(keep_cols))
)
print(dim(ppe_labeled))

write.csv(ppe_labeled, "./ppe_labeled.csv", row.names = FALSE)