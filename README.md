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


## Brief description of the scripts

This set of scripts is useful for automating the extraction, comparison, and optimization of model outcomes in data science. They navigate directories, and gather and organize data required for confusion matrices and classifier outputs. This includes analyzing such data to determine the best classifiers for ensemble methods, in this case voting classifiers. The scripts combine multiple Weka model predictions against a dataset, yielding superior combined results when compared to individual models.

These scripts use the Weka Evaluation class in their operational process to perform 10-fold cross-validation for each classifier. Serializing predictions and evaluation results for later use, as well as extracting and aggregating cross-validated predictions, probabilities, and evaluation results across folds, are all part of this process. The end result is a comprehensive set of combined prediction files that detail overall predictions per classifier as well as instance-specific predictions across classifiers.

Furthermore, these scripts calculate statistical measures that are essential for evaluating model performance. They can read data in a variety of formats, including serialized models and variables, JSON and CSV files. Their capabilities include calculating maximum values and combinations from results, used in determining the most effective voting ensemble method for each case. It enables statistics to be computed to compare the accuracy of ensemble methods versus individual classifier performance. Finally, they allow calculating the effectiveness of model combinations and identify opportunities for optimization, assisting in the decision-making process for selecting efficient ensemble methods in data analysis.

## Instructions to make the scripts work

The scripts are intended to work with pre-processed and parameterized Speech Emotion Recognition (SER) databases. This parameterization procedure entails preparing the databases and storing them as files in the **./arff** folder. To use these scripts effectively, users must first create an arff folder. Files containing parameterized databases should be placed in this folder. The databases were parameterized in our specific implementation using OpenSmile eGemaps, a command-line tool for feature extraction. The resulting **.arff** files from this parameterization are saved in the arff folder. The codes used for each emotion analyzed in the SER are described in detail in the associated article.

The scripts work assuming that SER databases have been previously parameterized and the parameterization is stored as a single file in a ./arff folder.
To make them work, create an arff folder and place a file or files with the parameterized databases. In our case, we parameterized them using OpenSmile eGemaps parameterization by command line scripting and placed the generated .arff files in the folder. Details on the codes used for each emotion are writtn in the article.

The scripts are named with digits at the beginning, in which the intended execution order is established from command line. 

Results are stored to specific folders, so the outcomes of each step are later used by the following scripts.

config.conf file defined, to specify paths for inputs (arff files locations, Weka .jar location) and outputs (results of the different scripts) throughout scripts when necessary. 

**Note:** Even though the script were developed for SER analysis, the scripts can be used for any kind of database, as long as they are in arff format.

**Note:** The scripts were developed in Python 2.
