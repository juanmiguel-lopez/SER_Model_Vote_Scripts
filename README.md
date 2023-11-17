# SER Model Vote Scripts

Scripts for Speech Emotion Recognition (SER)

## Related Publication

This code was developed in support of the research article titled:

"Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios"

which is currently under review in Expert Systems and Applications journal.

Please note that the article is not yet published, and the details provided are subject to change upon publication. The code in this repository reflects the methodologies and results as presented in the manuscript. Once the article is published, this section will be updated with the full citation and a link to the published work (if available).

## Citation

For now, if you use this code in your research or project, please cite it as follows:

Juan-Miguel López-Gil, Nestor Garay-Vitoria (2023) "Assessing the Effectiveness of Ensembles in Speech Emotion Recognition: Performance Analysis under Challenging Scenarios", Expert Systems with Applications (article submitted for publication)

**Note:** This citation will be updated once the article is published.


## Instructions to make the scripts work

The scripts work assuming that SER databases have been previously parameterized and the parameterization is stored as a single file in a ./arff folder.
To make them work, create an arff folder and place a file or files with the parameterized databases. In our case, we parameterized them using OpenSmile eGemaps parameterization by command line scripting and placed the generated .arff files in the folder. Details on the codes used for each emotion are writtn in the article.

The scripts are named with digits at the beginning, in which the intended execution order is established. 

Results are stored to specific folders, so the outcomes of each step are later used by the following scripts.

config.conf file defined, to specify paths for inputs and outputs throughout scripts when necessary.

Even though the script were developed for SER analysis, the scripts can be used for any kind of database, as long as they are in arff format.

The scripts were developed in Python 2.
