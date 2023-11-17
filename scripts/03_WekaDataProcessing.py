# Importing necessary libraries and modules
from __future__ import division

import os
import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random, from_commandline
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
import weka.core.serialization as serialization

import time
from multiprocessing import Process
import sys
import pandas as pd
import csv
import glob


def load_classifier_models(models_route):
    import glob
    import fnmatch
    
    matches = []
    for root, dirnames, filenames in os.walk(models_route):
        for filename in fnmatch.filter(filenames, '*.model'):
            matches.append(os.path.join(root, filename))
     
    return matches    
        
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
    default_classifier_values.append("weka.classifiers.trees.ZeroR")
    default_classifier_values.append("weka.classifiers.trees.DecisionStump")     
    default_classifier_values.append("weka.classifiers.trees.HoeffdingTree -L 2 -S 1 -E 1.0E-7 -H 0.05 -M 0.01 -G 200.0 -N 0.0")
    default_classifier_values.append("weka.classifiers.trees.J48 -C 0.25 -M 2")
    default_classifier_values.append("weka.classifiers.trees.LMT -I -1 -M 15 -W 0.0")     
    default_classifier_values.append("weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1")
    default_classifier_values.append("weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1")
    default_classifier_values.append("weka.classifiers.trees.REPTree -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0")
    
    return default_classifier_values

def get_classes_form_data_file(search_file):
    
    jvm.start(packages=True, class_path=['/home/juanmi/anaconda/lib/python2.7/site-packages/weka/lib/python-weka-wrapper.jar', '/home/juanmi/weka-3-8-2/weka.jar'])
    print('Java Bridge Loaded!')
    
    time.sleep(3)
    
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(search_file)
    data.class_is_last()
    
    classes_list = []
    for inst in data:
        print(inst)
        classes_list.append(int(inst.get_value(inst.class_index)+1))
    
    return classes_list

def evaluate_data_file(search_files, clf):
    
    jvm.start(packages=True, class_path=['/home/juanmi/anaconda/lib/python2.7/site-packages/weka/lib/python-weka-wrapper.jar', '/home/juanmi/weka-3-8-2/weka.jar'])
    print('Java Bridge Loaded!')
    
    time.sleep(3)
    
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(search_files)

    ninst = data.num_instances
    
    nattr = data.num_attributes
    filtered = data
    ninst = filtered.num_instances
    nattr = filtered.num_attributes
    
    filtered.class_is_last()
    evaluation = Evaluation(filtered)
    
    print('Reading ' + clf + ' model')
    objects = serialization.read_all(clf)
    classifier = Classifier(jobject=objects[0])
        
    algorithm_starting_time = int(round(time.time() * 1000))
    
    clf_predictions = list()
    clf_class_values = list()
    
    for inst in filtered:
        pred = evaluation.test_model_once(classifier, inst)
        clf_predictions.append(int(pred+1))
    
    predictions = ""
    for pred in clf_predictions:
        predictions += ',\'' + str(int(pred)) + '\'\n'
    
    t = clf.split('../',1)[1]
    t2 = t.split('_',1)[0]
            
    t3 = clf.split('all.arff/',2)[2]
    
    return clf_predictions
    
def trial1(data_path_line):
    evaluate_data_file(data_path_line, myfile)
    
    
def process_arff_file(search_file, clf_list, output_path_line):
  
    columns = []
    mylists = []
    
    r = search_file.split('../arffs/',1)[1]
    r2 = r.split('_all.arff')[0]
    
    classes_list = get_classes_form_data_file(search_file)
    
    for clf in clf_list:
        
        t = clf.split('../',1)[1]
        t2 = t.split('_all.arff',1)[0]
    
        if (r2 == t2):
            mylist = evaluate_data_file(search_file, clf)
            s = clf.split('_all.arff/',2)[2]
            s = s.split('.model',1)[0]
            columns.append(s)
            mylists.append(mylist)
            
    zipped_list_prev = zip(*mylists)
    zipped_list = [old + (new,) for old, new in zip(zipped_list_prev, classes_list)]
    columns.append('Class')
    
    #with open("output.csv", "wb") as f:
    with open(output_path_line + r2 + ".model.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(zipped_list)  # Add nested structure, multi line

    with open(output_path_line + r2 + ".model.columns.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(columns) 
    
    with open(output_path_line + r2 + ".all_together.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zipped_list)
            
    
if __name__ == "__main__":
    
    program = sys.argv[0]
    print("Program running is:", program)
    #Now check for extra arguments
    if (len(sys.argv) == 2):
        argument1 = sys.argv[1]
        print("Argument:", argument1)
        
    config_lines = []
    for line in open(argument1):
        li=line.strip()
        if not li.startswith("#"):
            config_lines.append(line.rstrip())
    config_lines[:] = [x for x in config_lines if x]
    
    arff_folder = config_lines[1]
    
    os.getcwd()
    
    if not os.path.exists(config_lines[6]):
        os.mkdir(config_lines[6])
    
    
    data_path_line = config_lines[2]
    output_path_line = config_lines[6]

    models_route = config_lines[5]
    
    
    clf_list = load_classifier_models(models_route)
    
    #csv_folder = "/media/juanmi/19EB39BF4E04DD28/tmp/CK_Plus/emaitzak_openface_CK_Plus/"
    arff_folder = config_lines[1]
    
    
    search_files_pattern =  arff_folder + "*.arff"
    search_files = glob.glob(search_files_pattern)
    
    processing_starting_time = int(round(time.time() * 1000))

    processes = list()
    
    for search_file in search_files:        
        #process_arff_file(search_file, clf_list, output_path_line)
        p = Process(target=process_arff_file, args=(search_file, clf_list, output_path_line))
        processes.append(p)
    
    # kick them off 
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()    
    
    processing_ending_time = int(round(time.time() * 1000))

    print("###  Overall Algorithm Processes Execution Time (ms):" + str(processing_ending_time - processing_starting_time))