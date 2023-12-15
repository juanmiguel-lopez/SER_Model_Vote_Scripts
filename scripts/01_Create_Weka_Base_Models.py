from __future__ import division

import os
import traceback
import weka.core.jvm as jvm
#import wekaexamples.helper as helper
from weka.core.classes import Random, from_commandline
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter

import weka.core.serialization as serialization

import time

from multiprocessing import Process


def load_default_classifier_values():
    
    default_classifier_values = list()
    
    default_classifier_values.append("weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5")
     
    default_classifier_values.append("weka.classifiers.bayes.NaiveBayes")
    default_classifier_values.append("weka.classifiers.bayes.NaiveBayesUpdateable")
    default_classifier_values.append("weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4")
    default_classifier_values.append("weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a")
    default_classifier_values.append("weka.classifiers.functions.SimpleLogistic -I 0 -M 500 -H 50 -W 0.0")
    default_classifier_values.append("weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\"")
    default_classifier_values.append("weka.classifiers.lazy.IBk -K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\"")
    default_classifier_values.append("weka.classifiers.lazy.KStar -B 20 -M a")
    default_classifier_values.append("weka.classifiers.lazy.LWL -U 0 -K -1 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\" -W weka.classifiers.trees.DecisionStump")
    default_classifier_values.append("weka.classifiers.rules.DecisionTable -X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\"")     
    default_classifier_values.append("weka.classifiers.rules.JRip -F 3 -N 2.0 -O 2 -S 1")
    default_classifier_values.append("weka.classifiers.rules.OneR -B 6")
    default_classifier_values.append("weka.classifiers.rules.PART -M 2 -C 0.25 -Q 1")
    default_classifier_values.append("weka.classifiers.trees.DecisionStump")
    default_classifier_values.append("weka.classifiers.trees.HoeffdingTree -L 2 -S 1 -E 1.0E-7 -H 0.05 -M 0.01 -G 200.0 -N 0.0")
    default_classifier_values.append("weka.classifiers.trees.J48 -C 0.25 -M 2")
    default_classifier_values.append("weka.classifiers.trees.LMT -I -1 -M 15 -W 0.0")
    default_classifier_values.append("weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1")
    default_classifier_values.append("weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1")
    default_classifier_values.append("weka.classifiers.trees.REPTree -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0")
    
    return default_classifier_values


def evaluate_data_file(data_path_line, myfile, clf, output_path_line, output_folder):
    
    jvm.start(packages=True, class_path=['../weka/weka.jar'])

    print('Java Bridge Loaded')

    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(myfile)
    data.class_is_last()

    #Classifier
    print("Selected classifier: " + clf)
    
    # perform cross-validation and add predictions
    evaluation = Evaluation(data)
    
    # cross-validate numeric classifier
    #classifier = Classifier(classname="weka.classifiers.functions.LinearRegression", options=["-S", "1", "-C"])
    cmdline = clf
    classifier = from_commandline(cmdline, classname="weka.classifiers.Classifier")
        
    algorithm_starting_time = int(round(time.time() * 1000))
    evaluation.crossvalidate_model(classifier, data, 10, Random(1))
    #Calculate overall time per algorithm
    algorithm_ending_time = int(round(time.time() * 1000))

    print(">>> " + clf + " algorithm execution time (ms): " + str(algorithm_ending_time - algorithm_starting_time))
    
    #Extract Accuracy
    mysummary = evaluation.percent_correct
    print(evaluation.summary())
    print(evaluation.class_details())
    print(evaluation.confusion_matrix)

    #Append Accuracy and Algorithm Settings to output
    with open(output_path_line + "weka_models_output.csv", "a+") as myfile:
        #myfile.write('\'' + mysummary + '\', \'' + cmdline + '\'\n')
        myfile.write('\'' + str(mysummary) + '\', \'' + str( (algorithm_ending_time - algorithm_starting_time) / 1000) + '\', \'' + cmdline + '\'\n')
        myfile.close()

    part1 = clf.split(' ', 1)[0]
    part2 = os.path.splitext(part1)[1]
    
    outfile = output_folder + part2[1:] + ".model"
    
    classifier.build_classifier(data)
    serialization.write(outfile, classifier)
    
    part1 = clf.split(' ', 1)[0]
    part2 = os.path.splitext(part1)[1]
    outfile = output_folder + part2[1:] + ".txt" 
    
    text_file = open(outfile, "w")
    text_file.write(str(evaluation.percent_correct)+'\n')
    text_file.write(str(evaluation.summary())+'\n')
    text_file.write(str(evaluation.class_details())+'\n')
    text_file.write(str(evaluation.matrix())+'\n')
    text_file.close()

    #Serialize Confussion Matrices        
    cf = evaluation.confusion_matrix       
    memfile = output_folder + part2[1:] + ".conf_mat"
    import numpy        
    #Serialize
    numpy.save(memfile, cf)
    
def main(argv):
        
    #Read config data for the database:
    config_lines = []
    #for line in open("./config/tuning_MMI.conf"):
    for line in open(argv):
        li=line.strip()
        if not li.startswith("#"):
            config_lines.append(line.rstrip())
    config_lines[:] = [x for x in config_lines if x]
    
    output_folder = config_lines[1]
    
    if not os.path.exists(output_folder):
        #os.mkdir(config_lines[5])
        os.mkdir(output_folder)

    output_folder2 = output_folder + config_lines[0] + "/"   
    if not os.path.exists(output_folder2):
        os.mkdir(output_folder2)

    output_folder3 = output_folder + config_lines[0]+"/"+config_lines[5]+config_lines[0] + "/"   
    if not os.path.exists(output_folder3):
        os.mkdir(output_folder3) 

    output_path_line = config_lines[1] + config_lines[0] + "/" + config_lines[4] + config_lines[0]  + "/"   
    
    if not os.path.exists(output_path_line):
        os.mkdir(output_path_line)       
        
    data_path_line = config_lines[2]
    
    myfile = config_lines[2] + config_lines[0]
    
    clf_list = load_default_classifier_values()
    
    for clf in clf_list:
        evaluate_data_file(data_path_line, myfile, clf, output_path_line, output_folder3)
        print("\n\n\n\nEnded " + clf + "\n\n\n")
    
    jvm.stop()

    print("Exiting")
    exit(0)

import sys
if __name__ == "__main__":
    main(sys.argv[1])
    exit(0)