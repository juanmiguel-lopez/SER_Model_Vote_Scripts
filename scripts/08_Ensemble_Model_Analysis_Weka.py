# Importing necessary libraries and modules
from __future__ import division
import numpy as np
import glob, os, numpy as np
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join
import traceback
import sys
from pathlib2 import Path
import json
import time

import weka.core.jvm as jvm
from weka.core.serialization import read
from weka.classifiers import Evaluation
import weka.core.serialization as serialization
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier
from weka.core.classes import Random, from_commandline
from weka.filters import Filter
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier
from weka.core.serialization import write

import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


def save_crossvalidated_instances_data(path_to_arffs, path_to_instances, path_to_save):
        
    path_to_arffs = '../arffs/'
    path_to_instances = './RESULTS_04_results_classifiers_crossvalidated/'
    
    dirName = './RESULTS_08_Ensembles_10_fold_results/'
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
    
    ##Nominal class (Linear Regression and so on)
    
    return default_classifier_values


def calculate_matches(clf_predictions_column, class_column):
    matches = 0
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
    
    num_rows, num_cols = accuracy_based.shape
    
    classifier_accuracies = clf_accuracies(accuracy_based, x, num_rows, num_cols)
    
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
            else:
                num_equals += 1
                
                maximum_value = -1
                
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
                        print('[ERROR] -> Strategy for equalities in classifier outcomes not recognized')
                        sys.exit(1)
        else:
            num_equals += 1
                
            maximum_value = -1
                
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
                    print('[ERROR] -> Strategy for equalities in classifier outcomes not recognized')
                    sys.exit(1)
            
        pred = int(maximum_value)
        predictions.append(pred)          
    

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

            
def main(myfile):
    
    current_path = os.getcwd()
    print(current_path)
    
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
    ii = list(np.nonzero(y)[0])
    instances_per_class = []
    for elem in ii:
        instances_per_class.append(y[elem])

    num_rows, num_cols = accuracy_based.shape
        
    conf_matrices = list()
    for clf in range(0, 21):
        column = list(accuracy_based[:,clf])
        classes = [[0 for i in range(0, len(instances_per_class))] for j in range(0, len(instances_per_class))]
        for current_position in range(0, len(column)):
            myrow = ii.index(x[current_position])
            mycolumn = ii.index(column[current_position])
            
            classes[myrow][mycolumn] += 1
        conf_matrices.append(classes)
        
    names = classifier_names()
    clfs = load_default_classifier_values()
     
    all_ensembles_per_combination = list()
    
    classifier_accuracies = clf_accuracies(accuracy_based, x, num_rows, num_cols)
    list1 = classifier_accuracies[0:len(classifier_accuracies)-1]
            
    filename = myfile[myfile.rfind('/')+1:]
    filename = filename.split('.')[0]
    
    dirName = current_path + '/RESULTS_08_Ensembles_10_fold_results/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    
    data_file = '../arffs/' + filename + '_all.arff'
    print("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()
        
    filename = data_file.split('.arff')[0]
    filename = filename.rsplit('/',1)[1]
    filename = filename.split('_all')[0]
    
    if not os.path.exists(dirName+filename):
        os.mkdir(dirName+filename)
        
    base_folder = dirName
    

    print("===> ===> ===> Starting with database: " + filename)
    
    for i in range(2, len(names)+1): #combinations, deleting worst one each iteration until the two best are left
        indexes_to_include = list()
        indexes_to_include = indexes_to_include_in_combination(list1, i, classifier_accuracies)
    
        classifiers = []
        
        for elem in indexes_to_include:
            classifiers.append(Classifier(classname=clfs[elem]))
        
        comb_name = ''
        for elem in indexes_to_include:
            comb_name += '_'+str(elem)
        
        new_folder = base_folder+filename+'/classifier_evaluations/'
        checkfilename = new_folder+filename+'_Vote'+comb_name+'.evaluations.txt'
        
        if not os.path.exists(checkfilename):
            combination_ensembles = list()
            
            vote_options = ['AVG', 'PROD', 'MAJ', 'MIN', 'MAX']
            #####vote_options = ['AVG']
            vote_metas = list()
            
            for elem in vote_options:
                vote_meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote", options=["-R", elem])
                vote_meta.classifiers = classifiers
                print(vote_meta.to_commandline())
                vote_metas.append(vote_meta)
                
                weka_string = 'java -Xmx32g -cp "/home/juanmi/weka-3-8-2/weka.jar" '
                weka_string += vote_meta.to_commandline()
                weka_string += ' -t ' + data_file
                weka_string += ' > '+new_folder+filename+'_Vote'+comb_name+'_'+elem+'.evaluations.txt'
                
                
                if os.path.exists('./SCRIPTS_08_4/'+filename+'.txt'):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not
                
                weka_file = open('./SCRIPTS_08_4/'+filename+'.txt', append_write)
                weka_file.write(weka_string + '\n')
                weka_file.close()
                
        print('===> Finished combination: ' + str(indexes_to_include))
      
    
    ending_time = int(round(time.time() * 1000))
    print("===> ===> ===> Finished with database: " + filename + " in: " + str(ending_time-starting_time) + " milliseconds")
        


def evaluate_folds_with_classifier(base_folder, filename, classifier, comb_name, data, num_folds, num_seed):

    folds = num_folds
    seed = num_seed
    
    rnd = Random(seed)
    rand_data = Instances.copy_instances(data)
    rand_data.randomize(rnd)
    if rand_data.class_attribute.is_nominal:
        rand_data.stratify(folds)
        
    predicted_data = None
    evaluation = Evaluation(rand_data)
    for i in range(folds):
        print('Fold: ' + str(i) + ', in file: ' + filename)
        train = rand_data.train_cv(folds, i, rnd)
        test = rand_data.test_cv(folds, i)

        cls = Classifier.make_copy(classifier)
        cls.build_classifier(train)
        evaluation.test_model(cls, test)

        addcls = Filter(
            classname="weka.filters.supervised.attribute.AddClassification",
            options=["-classification", "-distribution", "-error"])
        addcls.set_property("classifier", Classifier.make_copy(classifier))
        addcls.inputformat(train)
        addcls.filter(train)  # trains the classifier
        pred = addcls.filter(test)
        if predicted_data is None:
            predicted_data = Instances.template_instances(pred, 0)
        for n in range(pred.num_instances):
            predicted_data.add_instance(pred.get_instance(n))


    
    new_folder = base_folder+filename+'/classifier_instances/'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    write(new_folder+filename+'_'+comb_name+'.instances', predicted_data)
        
    myinstances = Instances(jobject=serialization.read(new_folder+filename+'_'+comb_name+'.instances'))
    print(myinstances)
    
    new_folder = base_folder+filename+'/classifier_evaluations/'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    filename = new_folder+filename+'_'+comb_name+'.evaluations.txt'
    
    text = ("\n=== Setup ===")
    text = text + ("\nClassifier: " + classifier.to_commandline())
    text = text + ("\nDataset: " + data.relationname)
    text = text + ("\nFolds: " + str(folds))
    text = text + ("\nSeed: " + str(seed))
    text = text + ("\n")
    
    print(text + (evaluation.summary("=== " + str(folds) + "-fold Cross-Validation ===")))
    evaluation_serialize_text_format(filename, text, evaluation)
        
    
    return predicted_data, evaluation


def evaluation_serialize_text_format(filename, text, evaluation):
    text = text + ("\n=== Summary ==="+evaluation.summary())
    text = text + ("\n"+evaluation.class_details())
    text = text + ("\n"+evaluation.matrix())
    text = text + ("\nareaUnderPRC/0: " + str(evaluation.area_under_prc(0)))
    text = text + ("\nweightedAreaUnderPRC: " + str(evaluation.weighted_area_under_prc))
    text = text + ("\nareaUnderROC/1: " + str(evaluation.area_under_roc(1)))
    text = text + ("\nweightedAreaUnderROC: " + str(evaluation.weighted_area_under_roc))
    text = text + ("\navgCost: " + str(evaluation.avg_cost))
    text = text + ("\ntotalCost: " + str(evaluation.total_cost))
    text = text + ("\nconfusionMatrix:\n " + str(evaluation.confusion_matrix))
    text = text + ("\ncorrect: " + str(evaluation.correct))
    text = text + ("\npctCorrect: " + str(evaluation.percent_correct))
    text = text + ("\nincorrect: " + str(evaluation.incorrect))
    text = text + ("\npctIncorrect: " + str(evaluation.percent_incorrect))
    text = text + ("\nunclassified: " + str(evaluation.unclassified))
    text = text + ("\npctUnclassified: " + str(evaluation.percent_unclassified))
    text = text + ("\ncoverageOfTestCasesByPredictedRegions: " + str(evaluation.coverage_of_test_cases_by_predicted_regions))
    text = text + ("\nsizeOfPredictedRegions: " + str(evaluation.size_of_predicted_regions))
    text = text + ("\nfalseNegativeRate: " + str(evaluation.false_negative_rate(1)))
    text = text + ("\nweightedFalseNegativeRate: " + str(evaluation.weighted_false_negative_rate))
    text = text + ("\nnumFalseNegatives: " + str(evaluation.num_false_negatives(1)))
    text = text + ("\ntrueNegativeRate: " + str(evaluation.true_negative_rate(1)))
    text = text + ("\nweightedTrueNegativeRate: " + str(evaluation.weighted_true_negative_rate))
    text = text + ("\nnumTrueNegatives: " + str(evaluation.num_true_negatives(1)))
    text = text + ("\nfalsePositiveRate: " + str(evaluation.false_positive_rate(1)))
    text = text + ("\nweightedFalsePositiveRate: " + str(evaluation.weighted_false_positive_rate))
    text = text + ("\nnumFalsePositives: " + str(evaluation.num_false_positives(1)))
    text = text + ("\ntruePositiveRate: " + str(evaluation.true_positive_rate(1)))
    text = text + ("\nweightedTruePositiveRate: " + str(evaluation.weighted_true_positive_rate))
    text = text + ("\nnumTruePositives: " + str(evaluation.num_true_positives(1)))
    text = text + ("\nfMeasure: " + str(evaluation.f_measure(1)))
    text = text + ("\nweightedFMeasure: " + str(evaluation.weighted_f_measure))
    text = text + ("\nunweightedMacroFmeasure: " + str(evaluation.unweighted_macro_f_measure))
    text = text + ("\nunweightedMicroFmeasure: " + str(evaluation.unweighted_micro_f_measure))
    text = text + ("\nprecision: " + str(evaluation.precision(1)))
    text = text + ("\nweightedPrecision: " + str(evaluation.weighted_precision))
    text = text + ("\nrecall: " + str(evaluation.recall(1)))
    text = text + ("\nweightedRecall: " + str(evaluation.weighted_recall))
    text = text + ("\nkappa: " + str(evaluation.kappa))
    text = text + ("\nKBInformation: " + str(evaluation.kb_information))
    text = text + ("\nKBMeanInformation: " + str(evaluation.kb_mean_information))
    text = text + ("\nKBRelativeInformation: " + str(evaluation.kb_relative_information))
    text = text + ("\nSFEntropyGain: " + str(evaluation.sf_entropy_gain))
    text = text + ("\nSFMeanEntropyGain: " + str(evaluation.sf_mean_entropy_gain))
    text = text + ("\nSFMeanPriorEntropy: " + str(evaluation.sf_mean_prior_entropy))
    text = text + ("\nSFMeanSchemeEntropy: " + str(evaluation.sf_mean_scheme_entropy))
    text = text + ("\nmatthewsCorrelationCoefficient: " + str(evaluation.matthews_correlation_coefficient(1)))
    text = text + ("\nweightedMatthewsCorrelation: " + str(evaluation.weighted_matthews_correlation))
    text = text + ("\nclass priors: " + str(evaluation.class_priors))
    text = text + ("\nnumInstances: " + str(evaluation.num_instances))
    text = text + ("\nmeanAbsoluteError: " + str(evaluation.mean_absolute_error))
    text = text + ("\nmeanPriorAbsoluteError: " + str(evaluation.mean_prior_absolute_error))
    text = text + ("\nrelativeAbsoluteError: " + str(evaluation.relative_absolute_error))
    text = text + ("\nrootMeanSquaredError: " + str(evaluation.root_mean_squared_error))
    text = text + ("\nrootMeanPriorSquaredError: " + str(evaluation.root_mean_prior_squared_error))
    text = text + ("\nrootRelativeSquaredError: " + str(evaluation.root_relative_squared_error))
    
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    
    text_file = open(filename,append_write)
    
    text_file.write(text)
    text_file.close()


# Main method: orchestrates the execution of the program
if __name__ == "__main__":
    
    try:
        program = sys.argv[0]
        print("Program running is:", program)
        if (len(sys.argv) == 2):
            myfile = sys.argv[1]
            print("Argument:", myfile)
        
            jvm.start(max_heap_size="32g")
            
            main(myfile)
        else:
            print('Number of arguments in correct, one is needed with a json file of analysed database')            
        
    except Exception:
        print(traceback.format_exc())
    finally:
        jvm.stop()        
    