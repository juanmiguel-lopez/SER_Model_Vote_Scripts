# SER Model Vote Scripts

Scripts for Speech Emotion Recognition (SER)

## Related Publication

This code was developed in support of the research article titled:

"Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios"

published in Expert Systems and Applications journal.


## Citation

Article citation:
Juan-Miguel López-Gil, Nestor Garay-Vitoria,
Assessing the effectiveness of ensembles in Speech Emotion Recognition: Performance analysis under challenging scenarios,
Expert Systems with Applications,
2023,
122905,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2023.122905.
(https://www.sciencedirect.com/science/article/pii/S0957417423034073)

**Note:** This citation will be updated once the article is published.

Code citation:
Juan-Miguel López-Gil (2023) Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios [Source Code]. https://doi.org/10.24433/CO.1788598.v1
DOI: 10.24433/CO.1788598.v1
https://codeocean.com/capsule/2894730/tree/v1


## Brief description of the scripts

This set of scripts is useful for automating the extraction, comparison, and optimization of model outcomes in data science. They navigate directories, and gather and organize data required for confusion matrices and classifier outputs. This includes analyzing such data to determine the best classifiers for ensemble methods, in this case voting classifiers. The scripts combine multiple Weka model predictions against a dataset, yielding superior combined results when compared to individual models.

These scripts use the Weka Evaluation class in their operational process to perform 10-fold cross-validation for each classifier. Serializing predictions and evaluation results for later use, as well as extracting and aggregating cross-validated predictions, probabilities, and evaluation results across folds, are all part of this process. The end result is a comprehensive set of combined prediction files that detail overall predictions per classifier as well as instance-specific predictions across classifiers.

Furthermore, these scripts calculate statistical measures that are essential for evaluating model performance. They can read data in a variety of formats, including serialized models and variables, JSON and CSV files. Their capabilities include calculating maximum values and combinations from results, used in determining the most effective voting ensemble method for each case. It enables statistics to be computed to compare the accuracy of ensemble methods versus individual classifier performance. Finally, they allow calculating the effectiveness of model combinations and identify opportunities for optimization, assisting in the decision-making process for selecting efficient ensemble methods in data analysis.

## Instructions to make the scripts work

The scripts are intended to work with pre-processed and parameterized Speech Emotion Recognition (SER) databases. This parameterization procedure entails preparing the databases and storing them as files in the **arff** folder. To use these scripts effectively, users must first create an arff folder. Files containing parameterized databases should be placed in this folder. The databases were parameterized in our specific implementation using OpenSmile, a command-line tool for feature extraction, with eGemaps acoustic parameter set. The resulting **.arff** files from this parameterization are saved in the arff folder. The codes used for each emotion analyzed in the SER are described in detail in the associated article. There is an **arff** folder in the repository with a sample dataset.

The script files are named numerically, indicating the order in which they should be run from the command line. This structured naming convention ensures a systematic and coherent workflow by guiding the user through each step of the process.

When the scripts are run, they generate results that are saved to specific folders. This organization ensures that the outputs of each script are easily accessible to the scripts that follow. This approach ensures that the output of each script is seamlessly integrated into the next stage of the analysis.

Additionally, the configuration of these scripts is managed through a **config.conf** file passed as an argument to the scripts from command line. This file is required because it specifies the paths for various inputs and outputs used during script execution. This includes the.arff file locations, the path to the Weka.jar file, and the directories where the results of various scripts are stored. The scripts provide a flexible interface for managing the various components of the data processing and analysis pipeline by centralizing these configurations in a single file. We have provided an example in **config.conf** in **scripts** folder, that needs to be customize for the specific execution environment. 
 

**Note:** Even though the script were developed for SER analysis, the scripts can be used for any kind of database, as long as they are in .arff format.

**Note:** The scripts were developed in Python 2.

**Note:** The Weka version used was 3.8.2.
