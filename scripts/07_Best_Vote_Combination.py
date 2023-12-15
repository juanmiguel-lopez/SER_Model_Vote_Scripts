###Combinations with different strategies based on the outcomes of base classifiers
###Outcome: one file per each Db, best ensemble accuracy, best classifier accuracy, and difference between them

import json
import glob
import time
import os

def calculate_maximum(mylist, max_value, max_i, max_j):
    for i in range(0, len(mylist)):
        for j in range(0, len(mylist[0])):
            if mylist[i][j][0]>max_value:
                max_value = mylist[i][j][0]
                max_i = i
                max_j = j
                flag = 'outcome'
    return max_value, max_i, max_j

def main():

    import sys

    config_lines = []
    for line in open(sys.argv[1]):
        li=line.strip()
        if not li.startswith("#"):
            config_lines.append(line.rstrip())
    config_lines[:] = [x for x in config_lines if x]


    #Load files with data
    current_path = config_lines[1] + config_lines[0]
    
    mypath = current_path + '/RESULTS_06_Combinations_outcome/'
    
    json_files = sorted(glob.glob(mypath+"*.json"))
    
    base_path = current_path + '/RESULTS_05_10_fold_results/'
    
    dirName = current_path + '/RESULTS_07_Best_Combinations/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        
    for i in range(0, len(json_files)/3):
        
        starting_time = int(round(time.time() * 1000))

        with open(json_files[i*3]) as json_file:
            combinations = json.load(json_file)
        with open(json_files[(i*3)+1]) as json_file:
            outcome = json.load(json_file)
        with open(json_files[(i*3)+2]) as json_file:
            outcome_no_majority = json.load(json_file)
        
        max_value1, max_i1, max_j1 = calculate_maximum(outcome, 0, -1, -1)
        max_value2, max_i2, max_j2 = calculate_maximum(outcome_no_majority, 0, -1, -1)
              
        flag = ''
        if max_value1>max_value2:
            flag = 'outcome'
            max_value, max_i, max_j = max_value1, max_i1, max_j1
        elif max_value1<max_value2:
            flag = 'outcome_no_majority'
            max_value, max_i, max_j = max_value2, max_i2, max_j2            
        else: #The same
            flag=' same_outcome_from_both'
            max_value, max_i, max_j = max_value1, max_i1, max_j1
            
        db_name = os.path.basename(os.path.normpath(json_files[i*3]))
        db_name = db_name.split('_combinations')[0]
        
        with open(dirName + db_name + ".txt", "w+") as outcomes_file:
            mytext = 'DB name: ' + db_name + '\nMax value: ' + str(max_value*100) \
                + '\nCombination: ' + str(max_i) + ' -> ' + str(combinations[0:max_i+2]) \
                + '\nStrategy: ' + str(max_j) + '\nMode: ' + flag
            #Parse file with base classifier results to get the best one
            with open(base_path + db_name + '.txt', "r") as base_clfs_file: 
                line = base_clfs_file.readline()
                max_accuracy = 0
                clf_name = ''
                while line:
                    value = float(line.split('with outcome: ', 1)[1])
                    if value>max_accuracy:
                        max_accuracy = value
                        clf_name = (line.split('model: ', 1)[1]).split(' ',1)[0]
                    line = base_clfs_file.readline()
                mytext += '\n\nBase Classifier with Maximum Accuracy: ' + clf_name + \
                    '\nAccuracy: ' + str(max_accuracy*100) + \
                    '\n\nEnsemble wins by: ' + str((max_value-max_accuracy)*100)
                    
            outcomes_file.write(mytext)
        
        ending_time = int(round(time.time() * 1000))
        
        print("===> ===> ===> Finished with database: " + db_name + " in: " + str(ending_time-starting_time) + " milliseconds")
        
if __name__ == "__main__":
    main()