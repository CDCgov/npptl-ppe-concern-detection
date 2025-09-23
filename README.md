# Description

Code to accompany Payne and Haas, “Detecting PPE concerns in OSHA complaints using machine learning to support infectious disease outbreak response” (submitted). 

## Repository organization

### Data 
A compressed directory containing raw data, `00_data-raw/`, is available with the initial release. Download this directory to the repository and decompress. All raw data used are publicly available:

- [CDC: Weekly Rates of Laboratory-Confirmed COVID-19 Hospitalizations from the COVID-NET Surveillance System](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Rates-of-Laboratory-Confirmed-COVID-19-Hosp/6jg4-xsqq/about_data)
- [U.S. Census Bureau: North American Industry Classification System (NAICS)](https://www.census.gov/naics/)
- [Department of Labor: OSHA](https://www.osha.gov/foia)

### Workflow

#### Setup Python environment
- Install conda.
- In your terminal, run `conda env create environment.yml -n npptl-ppe-concern-detection-env` to create the conda environment.
- In your terminal, run `conda activate npptl-ppe-concern-detection-env` to activate the conda environment. Ensure you activate the environment in any high-performance computing job scripts.
- In your terminal, type `libretranslate --load-only en,es` to run the LibreTranslate API server, which we use to locally translate Spanish complaints into English (note: there is no space between `en` and `es`). You should see "Running on http://127.0.0.1:5000." Run all subsequent code in a separate terminal window.

#### Clean data
Run scripts in `01_clean-data/` in the following order.
- `01_merge-naics.py` merges OSHA dataset with NAICS datasets to get descriptive industry sector titles.
- `02_detect-language.py` detects the language of all complaints.
- `03_translate-spanish-to-english.py` translates Spanish narratives into English. 
- `03_filter-ppe.py` identifies PPE-related complaints, i.e., those with narratives containing at least one keyword from `ppe_terms.csv`.
- `05_cleanup.r` performs additional cleaning tasks.

An R function is available `utils/helper_functions.r` to facilitate downstream data cleaning when reloading OSHA complaints data for processing and visualization in R.

#### Sample complaints
`sample-complaints.py` produces a sample of 3,200 complaints and a version of the sample retaining only complaints with distinct hazard narratives that will eventually constitute the labeled dataset.

#### PPE coding
At this point, the PPE concerns in distinct hazard narratives among sampled complaints would be manually labeled. See paper for details. The labeled dataset annotated by the authors and accompanying data dictionary are provided in `03_ppe-coding/` as `ml_dataset.csv` and `ml_dataset_data_dictionary.csv`, respectively. Descriptive statistics for PPE-related complaints and the ML dataset can be found in `descriptives.ipynb`.

#### Fine-tune DistilBERT model and generate out-of-sample predictions
Run scripts in `04_distilbert/` in the following order.

- `distilbert_sim.py`: This Python script fine-tunes a DistilBERT model on one train-test split of the the labeled dataset and records evaluation metrics. This script requires one integer argument specifying the seed to use for the train-test split (`python distilbert_sim.py --seed <SEED_NUMBER>`). We run this script as an array job with 150 tasks on a high performance computing cluster using the task number as the `--seed` argument, so that 150 simulations are performed using seeds 0-149. Within the array job script, be sure to activate the conda environment using `source activate osha-covid-github`. Also be sure to request 2 slots per task to match the Python script.

- `predict_oos_m[0-2].py`: Each of these prediction scripts uses one of the 150 previously fine-tuned models generate a set of out-of-sample predictions for all PPE-related complaints that were not among the 3,200 initially sampled. The three sets of out-of-sample predictions are combined via (entrywise) majority vote to generate one vector-valued label for each out-of-sample complaint.

#### Create tables and figures summarizing results
The directory `05_results/` contains code to produce the tables and figures in the paper.

- `process_results.r` processes ML results in preparation for plotting and analysis.
- `results_ml.ipynb` produces tables and figures summarizing the detection ability of the fine-tuned DistilBERT model.
- `results_trends.ipynb` produces trend analyses in the paper.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the CC0 1.0 Universal public domain dedication. All contributions to this repository will be released under the CC0 dedication. By submitting a pull request you are agreeing to comply with this waiver of copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under the terms of the Apache Software License version 2, or (at your option) any later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and information. All material and community participation is covered by the Disclaimer and Code of Conduct. For more information about CDC's privacy policy, please visit http://www.cdc.gov/other/privacy.html.

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by forking and submitting a pull request. (If you are new to GitHub, you might start with a basic tutorial.) By contributing to this project, you grant a world-wide, royalty-free, perpetual, irrevocable, non-exclusive, transferable license to all users under the terms of the Apache Software License v2 or later.

All comments, messages, pull requests, and other submissions received through CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at http://www.cdc.gov/other/privacy.html.

## Records Management Standard Notice
This repository is not a source of government records but is a copy to increase collaboration and collaborative potential. All government records will be published through the CDC web site.