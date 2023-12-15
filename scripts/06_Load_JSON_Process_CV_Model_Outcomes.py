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

import time

from weka.core.classes import Random, from_commandline

from weka.filters import Filter

from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier

from weka.core.serialization import write


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


def clf_accuracies(accuracy_based, x, num_rows, num_cols):
    classifier_accuracies = list()
    for column in range(0,num_cols):
        mycol = accuracy_based[:,column]
        matches = 0
        for i in range(0, len(mycol)):
            if mycol[i]==x[i]:
                matches += 1
        classifier_accuracies.append(matches/num_rows)
    
    return classifier_accuracies


def calculate_classifier_accuracy_based_results(accuracy_based, x, instances_per_class, strategy, conf_matrices, data, indexes_to_include, mode):
    
    #Strategy = 1 -> Minimum number of instances (unbalanced datasets)
    #Strategy = 2 -> Classifier SUM with biggest accuracy (balanced datasets)
    #Strategy = 3 -> Classifier SUM with biggest accuracy for the given outcome - 10-fold cv confusion matrix (balanced datasets)    
    #Strategy = 4 -> Biggest probability per predictions (0-1 per class) for current instance
    #Strategy = 5 -> Biggest chance per class (classifier that says class1 with biggest %)
    
    
    num_rows, num_cols = accuracy_based.shape
    
    
    #Strategy 2: Classifiers by accuracies
    classifier_accuracies = clf_accuracies(accuracy_based, x, num_rows, num_cols)
    
    #Strategy 3: Get confussion matrices ###and load the best accuracies per class, ordered
    #'RESULTS_04_results_classifiers_crossvalidated'
    
    #Calculate voting for all
    all_z = accuracy_based[:,0:num_cols]
    predictions = []
    
    num_equals = 0
    
    for z in all_z:
        items = np.unique(z, return_counts=True)
        p1 = items[1]
        result = np.where(p1 == np.amax(p1))
        
        if mode==1:
            if len(result[0]) == 1:
                maximum_value = items[0][result[0][0]]
            #maximum_position = items[1][result[0][0]]
            else:
                num_equals += 1
                
                maximum_value = -1
                
                #Choose the emotion with less instances for training
                emo_instances = [item[1] for item in instances_per_class]
                
                emo_min = sys.maxint
                emo_index = -1
                #i = 0
                for index in result[0]:
                    if strategy == 1:
                        if emo_instances[index]<emo_min:
                            emo_min = emo_instances[index]
                            emo_index = index
                        maximum_value = emo_index   
                    elif strategy == 2:
                        list_clfs = [0] * len(result[0])
                        for i in range(0, len(z)):
                            for j in range(0, len(list_clfs)):
                                if z[i]==items[0][result[0][j]]:
                                    list_clfs[j] += classifier_accuracies[i]
                        emo_value = 0 
                        for i in range(0, len(list_clfs)):
                            if list_clfs[i]>emo_value:
                                emo_index = result[0][i] 
                                emo_value = list_clfs[i]
                        maximum_value = items[0][emo_index]
                    elif strategy == 3:
                        list_clfs = [0] * len(result[0])
                        for i in range(0, len(z)):
                            for j in range(0, len(list_clfs)):
                                if z[i]==items[0][result[0][j]]:
                                    list_clfs[j] += conf_matrices[i][result[0][j]][result[0][j]]
                        emo_index = list_clfs.index(max(list_clfs))
                        maximum_value = items[0][emo_index]
                    elif strategy == 4:
                        #data[1][i][1]
                        max_prob = 0
                        emo_index = -1
                        for i in range(0, len(data[1])):
                            if i in indexes_to_include:
                                mylist = data[1][i][1][1]
                                for elem in mylist:
                                    if elem>max_prob:
                                        max_prob=elem
                                        emo_index=i
                        maximum_value = data[1][emo_index][1][0]
                    elif strategy == 5:
                        list_clfs = [0] * len(result[0])
                        for i in range(0, len(z)):
                            for j in range(0, len(list_clfs)):
                                if z[i]==items[0][result[0][j]]:
                                    d1 = conf_matrices[i][result[0][j]][result[0][j]]
                                    mypercent = d1/sum(conf_matrices[i][result[0][j]])
                                    list_clfs[j] += mypercent
                        emo_index = list_clfs.index(max(list_clfs))
                        maximum_value = items[0][emo_index]
                    else:
                        print('[ERROR] Strategy for equalities in classifier outcomes not recognized')
                        sys.exit(1)
        else:
            num_equals += 1
                
            maximum_value = -1
                
            #Choose the emotion with less instances for training
            emo_instances = [item[1] for item in instances_per_class]
                
            emo_min = sys.maxint
            emo_index = -1
            for index in result[0]:
                if strategy == 1:
                    if emo_instances[index]<emo_min:
                        emo_min = emo_instances[index]
                        emo_index = index
                    maximum_value = emo_index   
                elif strategy == 2:
                    list_clfs = [0] * len(result[0])
                    for i in range(0, len(z)):
                        for j in range(0, len(list_clfs)):
                            if z[i]==items[0][result[0][j]]:
                                list_clfs[j] += classifier_accuracies[i]
                    emo_value = 0 
                    for i in range(0, len(list_clfs)):
                        if list_clfs[i]>emo_value:
                            emo_index = result[0][i] 
                            emo_value = list_clfs[i]
                    maximum_value = items[0][emo_index]
                elif strategy == 3:
                    list_clfs = [0] * len(result[0])
                    for i in range(0, len(z)):
                        for j in range(0, len(list_clfs)):
                            if z[i]==items[0][result[0][j]]:
                                list_clfs[j] += conf_matrices[i][result[0][j]][result[0][j]]
                    emo_index = list_clfs.index(max(list_clfs))
                    maximum_value = items[0][emo_index]
                elif strategy == 4:
                    #data[1][i][1]
                    max_prob = 0
                    emo_index = -1
                    for i in range(0, len(data[1])):
                        if i in indexes_to_include:
                            mylist = data[1][i][1][1]
                            for elem in mylist:
                                if elem>max_prob:
                                    max_prob=elem
                                    emo_index=i
                    maximum_value = data[1][emo_index][1][0]
                elif strategy == 5:
                    list_clfs = [0] * len(result[0])
                    for i in range(0, len(z)):
                        for j in range(0, len(list_clfs)):
                            if z[i]==items[0][result[0][j]]:
                                d1 = conf_matrices[i][result[0][j]][result[0][j]]
                                mypercent = d1/sum(conf_matrices[i][result[0][j]])
                                list_clfs[j] += mypercent
                    emo_index = list_clfs.index(max(list_clfs))
                    maximum_value = items[0][emo_index]
                else:
                    print('[ERROR] Strategy for equalities in classifier outcomes not recognized')
                    sys.exit(1)
            
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

    return percentage, num_equals
 
    
def index_to_remove(list1, j, classifier_accuracies, indexes_to_remove):

    list1.sort() 
    if classifier_accuracies.index(list1[-j]) not in indexes_to_remove:
        return classifier_accuracies.index(list1[-j]) 
    else:
        #repeated_value = classifier_accuracies.index(list1[-j])
        repeated_index = classifier_accuracies.index(list1[-j])
        while repeated_index in indexes_to_remove:
            repeated_index  = classifier_accuracies.index(list1[-j], repeated_index+1, len(classifier_accuracies))
        return repeated_index
    
    
def indexes_to_include_in_combination(list1, index, classifier_accuracies):
    list1.sort(reverse=True) 
    
    indexes_to_include = list()
    for i in range(0, index):
        if classifier_accuracies.index(list1[i]) not in indexes_to_include:
            indexes_to_include.append(classifier_accuracies.index(list1[i])) 
        else:
            #repeated_value = classifier_accuracies.index(list1[-j])
            repeated_index = classifier_accuracies.index(list1[i])
            while repeated_index in indexes_to_include:
                repeated_index  = classifier_accuracies.index(list1[i], repeated_index+1, len(classifier_accuracies))
            indexes_to_include.append(repeated_index)
    
    return indexes_to_include

def vote_conf_list(comb_list):
    mylist = list()
    values_list = load_default_classifier_values()
    
    for i in range(0, len(comb_list)):
        local_str = values_list[comb_list[i]]
        local_str.replace('\\','\'')
        local_str.replace('\"','"')
        mylist.append(local_str)
    
    return mylist


def vote_configuration(conf_list):
   stacking_string = "weka.classifiers.meta.Vote -S 1"
   for elem in conf_list:
       stacking_string += " -B \"" + elem + "\""
   return stacking_string + " -R AVG"


def load_default_classifier_values():
    
    default_classifier_values = list()

    default_classifier_values.append("weka.classifiers.bayes.BayesNet")
    default_classifier_values.append("weka.classifiers.bayes.NaiveBayes")
    default_classifier_values.append("weka.classifiers.bayes.NaiveBayesUpdateable")     
    default_classifier_values.append("weka.classifiers.functions.Logistic")
    default_classifier_values.append("weka.classifiers.functions.MultilayerPerceptron")
    default_classifier_values.append("weka.classifiers.functions.SimpleLogistic")
    default_classifier_values.append("weka.classifiers.functions.SMO")    
    default_classifier_values.append("weka.classifiers.lazy.IBk")
    default_classifier_values.append("weka.classifiers.lazy.KStar")
    default_classifier_values.append("weka.classifiers.lazy.LWL")       
    default_classifier_values.append("weka.classifiers.rules.DecisionTable")
    default_classifier_values.append("weka.classifiers.rules.JRip")
    default_classifier_values.append("weka.classifiers.rules.OneR")
    default_classifier_values.append("weka.classifiers.rules.PART")
    default_classifier_values.append("weka.classifiers.trees.DecisionStump")
    default_classifier_values.append("weka.classifiers.trees.HoeffdingTree")
    default_classifier_values.append("weka.classifiers.trees.J48")
    default_classifier_values.append("weka.classifiers.trees.LMT")
    default_classifier_values.append("weka.classifiers.trees.RandomForest")
    default_classifier_values.append("weka.classifiers.trees.RandomTree")
    default_classifier_values.append("weka.classifiers.trees.REPTree")
    
    return default_classifier_values

            
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

    mypath = config_lines[1] + config_lines[0] + "/RESULTS_05_10_fold_results/"
    json_files = glob.glob(mypath+"*.json")
    
    for myfile in json_files:
        
        starting_time = int(round(time.time() * 1000))
    
        with open(myfile) as json_file:
            data = json.load(json_file)
        
        db = list()
        for j in range(1, len(data[1][0])):
            instance = list()
            
            for i in range(0, len(data[1])):
                instance.append(data[1][i][j][0])
         
            instance.append(data[1][i][j][2]) #Append instance class
            db.append(instance)
            
        accuracy_based = np.array(db).astype(int)
        
        x = accuracy_based[:,21]
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        instances_per_class = zip(ii,y[ii])
    
        num_rows, num_cols = accuracy_based.shape
            
        #Confussion matrices per each classifier of current DB
        conf_matrices = list()
        for clf in range(0, 21):
            column = accuracy_based[:,clf]
            classes = [[0 for i in range(0, len(instances_per_class))] for j in range(0, len(instances_per_class))]
            for current_position in range(0, len(column)):
                p1 = [item for item in instances_per_class if item[0] == x[current_position]]
                myrow = instances_per_class.index(p1[0])
                p1 = [item for item in instances_per_class if item[0] == column[current_position]]
                mycolumn = instances_per_class.index(p1[0])
                #p3 = instances_per_class[p2-1]
                classes[myrow][mycolumn] += 1
            conf_matrices.append(classes)
            
        names = classifier_names()
        clfs = load_default_classifier_values()
         
        all_results_per_combination = list()##([] for i in range(5))
        #######all_ensembles_per_combination = list()
        all_results_per_combination_2 = list()
        
        classifier_accuracies = clf_accuracies(accuracy_based, x, num_rows, num_cols)
        list1 = classifier_accuracies[0:len(classifier_accuracies)-1]
                
        filename = myfile[myfile.rfind('/')+1:]
        filename = filename.split('.')[0]
        
        ###dirName = current_path + '/RESULTS_06_Combinations_outcome/'
        dirName = config_lines[1] + config_lines[0] + '/RESULTS_06_Combinations_outcome/'

        if not os.path.exists(dirName):
            os.mkdir(dirName)
        
        if not os.path.exists(dirName+filename):
            os.mkdir(dirName+filename)
            
        base_folder = dirName
        
        print("===> ===> ===> Starting with database: " + filename)
        
        for i in range(2, len(names)+1): #combinations, deleting worst one each iteration until the two best are left
            #Gather classifier accuracies
            indexes_to_include = list()
            #list1 = classifier_accuracies[:]
            indexes_to_include = indexes_to_include_in_combination(list1, i, classifier_accuracies)
        
            #Our voting strategies
            comb_accuracy_based = accuracy_based[:, indexes_to_include]
            
            current_combination = [
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 1, conf_matrices, data, indexes_to_include, 1), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 2, conf_matrices, data, indexes_to_include, 1), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 3, conf_matrices, data, indexes_to_include, 1), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 4, conf_matrices, data, indexes_to_include, 1), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 5, conf_matrices, data, indexes_to_include, 1)
                ]
            
            current_combination_2 = [
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 1, conf_matrices, data, indexes_to_include, 2), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 2, conf_matrices, data, indexes_to_include, 2), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 3, conf_matrices, data, indexes_to_include, 2), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 4, conf_matrices, data, indexes_to_include, 2), 
                calculate_classifier_accuracy_based_results(comb_accuracy_based, x, instances_per_class, 5, conf_matrices, data, indexes_to_include, 2)
                ]
            
            
            all_results_per_combination.append(current_combination)
            all_results_per_combination_2.append(current_combination_2)
            
            print('===> Finished combination: ' + str(indexes_to_include))
          
        
        json.dump(all_results_per_combination, open(dirName + filename + '_outcome.json', 'w'))
        json.dump(all_results_per_combination_2, open(dirName + filename + '_outcome_no_majority.json', 'w'))
        json.dump(indexes_to_include, open(dirName + filename + '_combinations.json', 'w'))
        
        ending_time = int(round(time.time() * 1000))
        print("===> ===> ===> Finished with database: " + filename + " in: " + str(ending_time-starting_time) + " milliseconds")
        

if __name__ == "__main__":
    try:
        jvm.start(max_heap_size="32g")
        main()
    except Exception, e:
        print(traceback.format_exc())
    finally:
        jvm.stop()        
