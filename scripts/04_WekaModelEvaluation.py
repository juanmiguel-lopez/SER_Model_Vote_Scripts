# Importing necessary libraries and modules
import os
import traceback

import weka.core.jvm as jvm
from weka.core.classes import Random, from_commandline
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter

from weka.core.serialization import write

import weka.core.serialization as serialization
    

import numpy as np
import xlwt
import glob
import sys

def overall_crossvalidated_predictions_per_classifier(base_folder, filename, data, predictions):
    cv_outcomes = list()
    for j in range(0, len(predictions[0])):
        cv_current_outcomes = 0.0
        for i in range(0, len(predictions)):
            if (predictions[i][j] == predictions[i][len(predictions[0])-1]):
                cv_current_outcomes += 1.0
        cv_outcomes.append(cv_current_outcomes/data.num_instances)
    
    y=np.array([np.array(xi) for xi in cv_outcomes]) 
    savefile = os.path.join(base_folder, filename+'_overall_crossvalidated_predictions_per_classifier')
    np.save(savefile, y)
    
    return cv_outcomes

    
def crossvalidated_predictions_per_file(base_folder, filename, instance_class, predicted_class):
    predictions = list()
    for j in range(0, len(instance_class[0])):
        current_instance_prediction = list()
        for i in range(0, len(instance_class)):
            current_instance_prediction.append(predicted_class[i][j])
        current_instance_prediction.append(instance_class[i][j])
        predictions.append(current_instance_prediction)
    
    y=np.array([np.array(xi) for xi in predictions])       
    savefile = os.path.join(base_folder, filename+'crossvalidated_predictions_per_file')
    np.save(savefile, y)
    
    return predictions
    
def crossvalidated_probabilities_per_classifier_per_instance(base_folder, filename, data, predictions):
    
    instance_class = list()
    predicted_class = list()
    probabilities_class = list()
    
    num_class_values = data.attribute_stats(data.num_attributes-1).distinct_count
    
    for i in range(0, len(predictions)):
        current_instance_class = list()
        current_predicted_class = list()
        current_probabilities_class = list()
        
        for j in range(0, predictions[0].num_instances):
            current_instance_class.append(predictions[i].get_instance(j).get_value(data.num_attributes-1))
            current_predicted_class.append(predictions[i].get_instance(j).get_value(data.num_attributes))

            local_probabilities = list()
            
            for i in range(1, num_class_values+1):
                local_probabilities.append(predictions[i].get_instance(j).get_value(data.num_attributes+i))
                
            current_probabilities_class.append(local_probabilities)
        
        instance_class.append(current_instance_class)
        predicted_class.append(current_predicted_class)
        probabilities_class.append(current_probabilities_class)
    
    y=np.array([np.array(xi) for xi in probabilities_class]) 
    savefile = os.path.join(base_folder, filename+'_crossvalidated_probabilities_per_classifier_per_instance')
    np.save(savefile, y)
    
    return instance_class, predicted_class, probabilities_class
    

def predictions_and_evaluations(base_folder, filename, base_classifiers, base_classifiers_text, data):
    
    predictions = list()
    evaluations = list()

    i = 0    
        
    for classifier in base_classifiers:
        
        classifier_name = base_classifiers_text[i].split('.')[3]
        classifier_name = classifier_name.split(' ')[0]

        predicted_data, evaluation = evaluate_folds_with_classifier(base_folder, filename, classifier_name, classifier, data, 10, 1)
        predictions.append(predicted_data)
        evaluations.append(evaluation) 
        
        i+=1

    return predictions, evaluations
    
        
def load_base_classifiers(default_classifier_values):
    base_classifiers = list()
    
    for elem in default_classifier_values:
        base_classifiers.append(from_commandline(elem, classname="weka.classifiers.Classifier"))

    return base_classifiers


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
    text = text + ("\nconfusionMatrix: " + str(evaluation.confusion_matrix))
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
    
    text_file = open(filename, "w")
    text_file.write(text)
    text_file.close()

    
#def evaluate_folds_with_classifier(base_folder, filename, classifier_name, classifier, data, 10, 1):
def evaluate_folds_with_classifier(base_folder, filename, classifier_name, classifier, data, num_folds, num_seed):

    folds = num_folds
    seed = num_seed
    
    rnd = Random(seed)
    rand_data = Instances.copy_instances(data)
    rand_data.randomize(rnd)
    if rand_data.class_attribute.is_nominal:
        rand_data.stratify(folds)
        
    # perform cross-validation and add predictions
    predicted_data = None
    evaluation = Evaluation(rand_data)
    for i in xrange(folds):
        train = rand_data.train_cv(folds, i, rnd)
        test = rand_data.test_cv(folds, i)

        # build and evaluate classifier
        cls = Classifier.make_copy(classifier)
        cls.build_classifier(train)
        evaluation.test_model(cls, test)

        # add predictions
        addcls = Filter(
            classname="weka.filters.supervised.attribute.AddClassification",
            options=["-classification", "-distribution", "-error"])
        # setting the java object directory avoids issues with correct quoting in option array
        addcls.set_property("classifier", Classifier.make_copy(classifier))
        addcls.inputformat(train)
        addcls.filter(train)  # trains the classifier
        pred = addcls.filter(test)
        if predicted_data is None:
            predicted_data = Instances.template_instances(pred, 0)
        for n in xrange(pred.num_instances):
            predicted_data.add_instance(pred.get_instance(n))

    new_folder = base_folder+'/classifier_instances/'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    #Serialize Predictions per model (class Instances)
    write(new_folder+filename+'_'+classifier_name+'.instances', predicted_data)

    #Serialize Evaluation per model (class Evaluation)
    myinstances = Instances(jobject=serialization.read(new_folder+filename+'_'+classifier_name+'.instances'))
    print(myinstances)
    
    new_folder = base_folder+'/classifier_evaluations/'
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    
    filename = new_folder+filename+'_'+classifier_name+'.evaluations.txt'
    
    text = ("\n=== Setup ===")
    text = text + ("\nClassifier: " + classifier.to_commandline())
    text = text + ("\nDataset: " + data.relationname)
    text = text + ("\nFolds: " + str(folds))
    text = text + ("\nSeed: " + str(seed))
    text = text + ("\n")
    
    print(text + (evaluation.summary("=== " + str(folds) + "-fold Cross-Validation ===")))
    #Serialize Evaluations per model -> Text file
    evaluation_serialize_text_format(filename, text, evaluation)
        
    return predicted_data, evaluation
    

def main():
    
    program = sys.argv[0]
    print("Program running is:", program)
    #Now check for extra arguments
    if (len(sys.argv) == 2):
        data_file = sys.argv[1]
        print("Argument:", data_file)
        
    print("Loading dataset: " + data_file)
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(data_file)
    data.class_is_last()

    filename = data_file.split('.arff')[0]
    filename = filename.rsplit('/',1)[1]
    filename = filename.split('_all')[0]
    
    base_folder = './RESULTS_04_results_classifiers_crossvalidated/'
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    base_folder = base_folder+filename
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    
    base_classifiers_text = load_default_classifier_values()
    base_classifiers = load_base_classifiers(base_classifiers_text)
    
    predictions, evaluations = predictions_and_evaluations(base_folder, filename, base_classifiers, base_classifiers_text, data)
            
    instance_class, predicted_class, probabilities_class = crossvalidated_probabilities_per_classifier_per_instance(base_folder, filename, data, predictions)
                
    predictions = crossvalidated_predictions_per_file(base_folder, filename, instance_class, predicted_class)
            
    cv_outcomes = overall_crossvalidated_predictions_per_classifier(base_folder, filename, data, predictions)
                
    print('Process complete')
    
    
if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception, e:
        print(traceback.format_exc())
        jvm.stop()
    finally:
        jvm.stop()
