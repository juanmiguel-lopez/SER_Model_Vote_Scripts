# Importing necessary libraries and modules
from __future__ import division
import numpy as np
import glob, os, numpy as np
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join
import sys
from pathlib2 import Path
import json
import traceback
import csv
import time


import weka.core.jvm as jvm
from weka.core.serialization import read
from weka.classifiers import Evaluation
import weka.core.serialization as serialization
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier



def save_crossvalidated_instances_data(path_to_arffs, path_to_instances, path_to_save):
    path_to_arffs = '../arffs/'
    path_to_instances = './RESULTS_04_results_classifiers_crossvalidated/'
    
    dirName = './RESULTS_05_10_fold_results/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
    clf_names = classifier_names()
    
    onlyfiles = [f for f in listdir(path_to_arffs) if isfile(join(path_to_arffs, f))]
        
    all_structure = list()
    
    for data_file in onlyfiles:
        
        print("Loading dataset: " + data_file)
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(path_to_arffs + data_file)
        data.class_is_last()
    
        
        class_column = []
        for elem in data:
            number = int(elem.get_string_value(elem.class_index))
            class_column.append(number)
        
        filename = data_file.split('.arff')[0]
        
        filename = filename.split('_all')[0]
        
        outcomes_file = open(dirName + filename + ".txt", "w")
        num_class_values = data.attribute_stats(data.num_attributes-1).distinct_count
    
        all_list = list()
        all_list.append(filename)
        
        clf_structure = list()
        
        
        for clf in clf_names:
            
            clf_list = list()
            
            clf_list.append(clf)
            
            f =  path_to_instances + filename + '/classifier_instances/' + filename + '_' + clf + '.instances'
            if Path(f).is_file():
                myinstances = Instances(jobject=serialization.read(f))
                matches = 0
                
                instance_number = 0
                
                for elem in myinstances:
                    instance_list = list()
                    instance_list.append(int(elem.get_string_value(data.num_attributes)))
                    
                    probabilities = list()
                    for i in range(1, num_class_values+1):
                        probabilities.append(elem.get_value(data.num_attributes+i))
                    instance_list.append(probabilities)
                    
                    
                    if elem.get_string_value(elem.num_attributes-1) == 'no':
                        matches += 1
                    instance_list.append(class_column[instance_number])
                    clf_list.append(instance_list)
                    
                instance_number += 1
            
                mytext = (filename + ' model: ' + clf + ' with outcome: ' + str(matches/myinstances.num_instances) + '\n')
            
            else:
                mytext = (filename + ' model: ' + clf + ' DOES NOT EXIST!!!\n')
    
            outcomes_file.write(mytext)
            
            clf_structure.append(clf_list)
            
        all_list.append(clf_structure) 
        all_structure.append(all_list)  
    
        json.dump(all_list, open(dirName + filename + '.json', 'w'))
        
    return all_structure
    
def classifier_names():
    
    default_classifier_values = list()    
    default_classifier_values.append("BayesNet")
    default_classifier_values.append("NaiveBayes")
    default_classifier_values.append("NaiveBayesUpdateable")
    default_classifier_values.append("Logistic")
    default_classifier_values.append("MultilayerPerceptron")
    default_classifier_values.append("SimpleLogistic")
    default_classifier_values.append("SMO")
    default_classifier_values.append("IBk")
    default_classifier_values.append("KStar")
    default_classifier_values.append("LWL")
    default_classifier_values.append("DecisionTable")
    default_classifier_values.append("JRip")
    default_classifier_values.append("OneR")
    default_classifier_values.append("PART")
    default_classifier_values.append("DecisionStump")
    default_classifier_values.append("HoeffdingTree")
    default_classifier_values.append("J48")
    default_classifier_values.append("LMT")
    default_classifier_values.append("RandomForest")
    default_classifier_values.append("RandomTree")
    default_classifier_values.append("REPTree")
    
    return default_classifier_values

def calculate_matches(clf_predictions_column, class_column):
    matches = 0
    
    for i in range(0, len(clf_predictions_column)):
        if (class_column[i] == clf_predictions_column[i]):
            matches+=1

    return matches
    
def calculate_classifier_accuracy_based_results(accuracy_based, x, instances_per_class):
    
    all_z = accuracy_based[:,0:20]
    predictions = []
    for z in all_z:
        items = np.unique(z, return_counts=True)
        p1 = items[1]
        result = np.where(p1 == np.amax(p1))
        
        if len(result[0]) == 1:
            maximum_value = items[0][result[0][0]]
            
        else:
            print("[WARNING] ---> Choose your strategy")            
            maximum_value = -1
            emo_instances = [item[1] for item in instances_per_class]
            
            emo_min = sys.maxint
            emo_index = -1
            i = 0
            for index in result[0]:
                current_index = items[0][result[0]][i] - 1
                if emo_instances[current_index]<emo_min:
                    emo_min = emo_instances[current_index]
                    emo_index = current_index + 1
                    
                i += 1    
            maximum_value = emo_index        
            
        pred = int(maximum_value)
        predictions.append(pred)          
    
    num_rows, num_cols = all_z.shape
    
    matches = 0
    for index in range(len(predictions)):
        if predictions[index] == x[index]:
            matches += 1

    percentage = matches/num_rows
    
    return percentage
        
def main():
    
    start = int(round(time.time() * 1000))
    save_crossvalidated_instances_data('', '', '')
    end = int(round(time.time() * 1000))
    
    print('TIME: ' + str(end-start))
    
    sys.exit()
    
    # Getting the current working directory
    current_path = os.getcwd()
    print(current_path)
        
    mypath = current_path + "/RESULTS_04_results_classifiers_crossvalidated/"
    
    directories = glob.glob(mypath+"*/")
    
    for mydir in directories:
        
        percentages = list()
        percentages.append(calculate_classifier_accuracy_based_results(accuracy_based, x, instances_per_class))
        
        names = classifier_names()
        
        combinations = ['p90', 'p75', 'median', 'mean']
        
        dirname = os.path.dirname(filename)
        dirname = os.path.basename(dirname)
    
        comb_file = os.path.join(current_path + '/results/', dirname + '_predictions.npy')
        combinations_data = np.load(comb_file)
        
        classifier_columns_names = os.path.join(current_path + '/results_classifiers/', dirname + '.model.columns.csv')

        with open(classifier_columns_names, "rb") as f:
            reader = csv.reader(f)
            classifier_columns = reader.next()
        
        for elem in combinations_data:               
            for elem2 in elem:
                
                common_positions = [classifier_columns.index(x) for x in elem2]
                all_pos = [i for i in range(0, len(classifier_columns))]           
                all_minus_commons = [i for i in range(0, len(all_pos)) if i not in common_positions]
                all_minus_commons.pop()                
                
                ac = np.delete(accuracy_based, all_minus_commons, axis=1)
                percentages.append(calculate_classifier_accuracy_based_results(ac, x, instances_per_class))
            
        print('Voting all completed, with accuracy: ' + str(percentage))
        
if __name__ == "__main__":
    try:
        jvm.start(max_heap_size="32g")
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()        
