# SER Model Vote Scripts

Scripts for Speech Emotion Recognition (SER)

## Related Publication

This code was developed in support of the research article titled:

"Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios"

which is currently under review in Expert Systems and Applications journal.

Please note that the article is not yet published, and the details provided are subject to change upon publication. The code in this repository reflects the methodologies and results as presented in the manuscript. Once the article is published, this section will be updated with the full citation and a link to the published work (if available).

## Citation

For now, if you use this code in your research or project, please cite it as follows:

Juan-Miguel López-Gil, Nestor Garay-Vitoria. "Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios", Expert Systems with Applications (article submitted for publication)

**Note:** This citation will be updated once the article is published.


## Instructions to make the scripts work

It automates the extraction and comparison of model outcomes. 
The scripts automatically navigate through these directories, collecting and organizing data pertinent to confusion matrices and classifier outputs. They analyze this data to select optimal classifiers for ensemble methods like voting systems.
the script combines multiple Weka model predictions against a dataset to produce superior combined results compared to individual models.
For each classifier, it runs 10-fold cross-validation using the Weka Evaluation class. The predictions and evaluation results are serialized for later use. The cross-validated predictions, probabilities, and evaluation results are extracted and aggregated across the folds. Combined prediction files are output - overall predictions per classifier, and predictions for each instance across classifiers. 
calculating statistical measures related to model performance.
The script's functionality includes loading data from JSON files, and CSV files

The scripts work assuming that SER databases have been previously parameterized and the parameterization is stored as a single file in a ./arff folder.
To make them work, create an arff folder and place a file or files with the parameterized databases. In our case, we parameterized them using OpenSmile eGemaps parameterization by command line scripting and placed the generated .arff files in the folder. Details on the codes used for each emotion are writtn in the article.

The scripts are named with digits at the beginning, in which the intended execution order is established. 

Results are stored to specific folders, so the outcomes of each step are later used by the following scripts.

config.conf file defined, to specify paths for inputs and outputs throughout scripts when necessary.

Even though the script were developed for SER analysis, the scripts can be used for any kind of database, as long as they are in arff format.

**Note:** The scripts were developed in Python 2.
