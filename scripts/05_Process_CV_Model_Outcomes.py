from __future__ import division

import numpy as np

import glob, os, numpy as np
from numpy import genfromtxt

from os import listdir
from os.path import isfile, join

import weka.core.jvm as jvm

from weka.core.serialization import read
from weka.classifiers import Evaluation
import weka.core.serialization as serialization

import traceback

from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier

import sys

from pathlib2 import Path

import json


def save_crossvalidated_instances_data(path_to_arffs, path_to_instances, path_to_save):
        
    dirName = path_to_save
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
    clf_names = classifier_names()
    
    onlyfiles = [f for f in listdir(path_to_arffs) if isfile(join(path_to_arffs, f))]
        
    all_structure = list()
    
    for data_file in onlyfiles:
        #Load file
        print("Loading dataset: " + data_file)
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(path_to_arffs + data_file)
        data.class_is_last()
    
        #Class column
        class_column = []
        for elem in data:
            number = int(elem.get_string_value(elem.class_index))
            class_column.append(number)
        
        ####### stringize data 
        str_data = list()
        for elem in data:
            p1 = str(elem)
            p2 = p1.rsplit(',',1)[0]
            str_data.append(str(p2))
        
        filename = data_file.split('.arff')[0]
        filename = filename.split('_all')[0]
        
        outcomes_file = open(dirName + filename + ".txt", "w")
        num_class_values = data.attribute_stats(data.num_attributes-1).distinct_count
    
        all_list = list()
        all_list.append(filename)
        
        clf_structure = list()
        
        #Load instances for classes classes
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
                    
                    ###Corresponding position in data for current instance
                    p1 = str(elem)
                    #p2 = p1.rsplit(',',num_class_values+2)[0]
                    p3 = p1.rsplit(',',num_class_values+3)[0]
                    #my_index = str_data.index(p2)
                    my_index = str_data.index(p3)
                    
                    #instance_list.append(class_column[instance_number])
                    instance_list.append(class_column[my_index])
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
    #[matches+=1 for i in range(0, len(pred)) if (x[i] == pred[i])]
    for i in range(0, len(clf_predictions_column)):
        if (class_column[i] == clf_predictions_column[i]):
            matches+=1

    return matches
    
def calculate_classifier_accuracy_based_results(accuracy_based, x, instances_per_class):
    
    #Calculate voting for all
    all_z = accuracy_based[:,0:20]
    predictions = []
    for z in all_z:
        items = np.unique(z, return_counts=True)
        p1 = items[1]
        result = np.where(p1 == np.amax(p1))
        
        if len(result[0]) == 1:
            maximum_value = items[0][result[0][0]]
            #maximum_position = items[1][result[0][0]]
        else:
            print("... [CAREFUL]!!! ---> Choose your strategy")
            
            maximum_value = -1
            
            #Choose the emotion with less instances for training
            emo_instances = [item[1] for item in instances_per_class]
            
            import sys
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
    

    #Calculate percentage for voting all data
    #x -> class, predicions -> clf_prediction    
    num_rows, num_cols = all_z.shape
    
    matches = 0
    for index in range(len(predictions)):
        if predictions[index] == x[index]:
            matches += 1

    percentage = matches/num_rows
        
    #all_matches = [calculate_matches(all_z[:,i], x) for i in range(0, num_cols)]
    return percentage
        
def main():
    
    import sys

    config_lines = []
    for line in open(sys.argv[1]):
        li=line.strip()
        if not li.startswith("#"):
            config_lines.append(line.rstrip())
    config_lines[:] = [x for x in config_lines if x]

    current_path = config_lines[1] + config_lines[0]
    
    mypath = config_lines[1] + config_lines[0] + "/RESULTS_04_results_classifiers_crossvalidated/"
    
    path_to_arffs = config_lines[2]
    path_to_instances = config_lines[1] + config_lines[0] + "/RESULTS_04_results_classifiers_crossvalidated/"

    path_to_save = config_lines[1] + config_lines[0] + "/RESULTS_05_10_fold_results/"
    
    import time
    start = int(round(time.time() * 1000))
    ###save_crossvalidated_instances_data('', '', '')
    save_crossvalidated_instances_data(path_to_arffs, path_to_instances, path_to_save)
    end = int(round(time.time() * 1000))
    
    print('TIME: ' + str(end-start))
    
    sys.exit()
    
    directories = glob.glob(mypath+"*/")
    print(directories)
    
    #Load one evaluation file, see it it has the data
    #trialfile = directories[0]+"classifier_evaluations/BERLIN_GER_BayesNet.evaluations"    
    #model = Classifier(jobject=serialization.read(outfile))
    
    
    for mydir in directories:
        percentages = list()
        percentages.append(calculate_classifier_accuracy_based_results(accuracy_based, x, instances_per_class))
        
        names = classifier_names()
        
        combinations = ['p90', 'p75', 'median', 'mean']
        
        #t = elem.split('weka_output_',1)[1]
        dirname = os.path.dirname(filename)
        dirname = os.path.basename(dirname)
    
        comb_file = os.path.join(current_path + '/results/', dirname + '_predictions.npy')
        combinations_data = np.load(comb_file)
        
        classifier_columns_names = os.path.join(current_path + '/results_classifiers/', dirname + '.model.columns.csv')
        import csv
        with open(classifier_columns_names, "rb") as f:
            reader = csv.reader(f)
            classifier_columns = reader.next()
        
        #For each possible percentile combination: (no, 90, 75, 50)
        for elem in combinations_data:   
            
            for elem2 in elem:
                
                common_positions = [classifier_columns.index(x) for x in elem2]
                all_pos = [i for i in range(0, len(classifier_columns))]           
                all_minus_commons = [i for i in range(0, len(all_pos)) if i not in common_positions]
                all_minus_commons.pop()
                
                #ac = np.delete(accuracy_based,np.s_[1:3],axis=1)
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