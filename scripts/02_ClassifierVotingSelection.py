# Importing necessary libraries and modules
import glob, os, numpy as np
import csv

# Getting the current working directory
current_path = os.getcwd()
print(current_path)

# Getting the current working directory
path = os.path.dirname(os.getcwd())
print(path)

# Listing all subfolders in the specified directory
list_subfolders_with_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

# Filtering directories that contain '.arff' files
list_subfolders_with_paths = [mydir for mydir in list_subfolders_with_paths if mydir.endswith('.arff')==True]

confussion_matrices = []
classifier_outputs = []
# Iterating through each directory to process the data
for mydir in list_subfolders_with_paths:
    matrices_folders = [x[0] for x in os.walk(mydir)]
    for elem in matrices_folders:
        if 'output' in elem:
            classifier_outputs.append(elem+'/weka_models_output.csv')

for elem in classifier_outputs:
    with open(elem) as csv_file:            
        csv_reader = csv.reader(csv_file, delimiter=',')
        outcomes = []
        categories = []
        for row in csv_reader:
            myrow = row[0]
            myrow = myrow.replace('\'','')
            outcomes.append(float(myrow))
            myrow = row[2]
            myrow = myrow.replace('\'','')
            categories.append(myrow)
        
    outcomes_categories = []
    for elem2 in categories:
        new_text = elem2.split("weka.classifiers.",1)[1]
        new_text = new_text.split(".",1)[1]
        new_text = new_text.split(" ",1)[0]
        outcomes_categories.append(new_text)
        
    y = np.array([np.array(xi) for xi in outcomes])   
    npa2 = np.argsort(y) 
    t = elem.split('weka_output_',1)[1]
    t2 = t.split('_all',1)[0]
    print(t2)
    
    
    print(np.amin(y))
    print(np.amax(y))
    print(np.std(y))
    print(np.var(y))

    statistics = []
    
    
    statistics.append(np.percentile(y, 90))
    statistics.append(np.percentile(y, 75))
    statistics.append(np.median(y))
    statistics.append(np.mean(y))
    

    statistics_classifiers = []
    
    for h in range(0, len(statistics)):    
        print('\nClassifiers for statistics[' + str(h) + ']')
        current_list = []
        for i in reversed(range(0, len(y))):
            if y[npa2[i]] > statistics[h]:
                print('Classifier: ' + str(outcomes_categories[npa2[i]]) + ', with accuracy: ' + str(y[npa2[i]]))
                current_list.append(str(outcomes_categories[npa2[i]]))
        statistics_classifiers.append(current_list)
        
    y=np.array([np.array(xi) for xi in statistics_classifiers])       
    
    savefile = os.path.join(current_path + '/results/', t2)
    np.save(savefile, y)
    
    print('-----------------------')

myfile = os.path.join(current_path + '/results/', 'classifiers.npy')
classifiers = np.load(myfile)

os.chdir(path + '/03_scripts/results/')

for myfile2 in glob.glob("weka*_all.arff.npy"):
    myfile = os.path.join(current_path + '/results/', myfile2)
    conf_matrix = np.load(myfile)
    num_rows, num_cols = conf_matrix.shape        
    
    view=''
    for i in range(0, num_rows):
        view += ''
    
    npa = np.sort(conf_matrix,axis=1) 
    npa2 = np.argsort(conf_matrix, axis=1) 
    
    t = myfile.split('weka_predictions_',1)[1]
    t2 = t.split('_all',1)[0]
    print(t2)    
    print(npa)
    
    overall_list = []
    
    for i in range(0, len(conf_matrix)):
        current_emotion_outcomes = conf_matrix[i]
        
        print(np.amin(current_emotion_outcomes))
        print(np.amax(current_emotion_outcomes))
        print(np.std(current_emotion_outcomes))
        print(np.var(current_emotion_outcomes))
    
        statistics = []
        
        statistics.append(np.percentile(current_emotion_outcomes, 90))
        statistics.append(np.percentile(current_emotion_outcomes, 75))
        statistics.append(np.median(current_emotion_outcomes))
        statistics.append(np.mean(current_emotion_outcomes))
    
        statistics_classifiers = []
        
        for h in range(0, len(statistics)):    
            print('\nClassifiers for statistics[' + str(h) + ']')
            current_list = []
            for j in reversed(range(0, len(current_emotion_outcomes))):
                if current_emotion_outcomes[npa2[i][j]] > statistics[h]:
                    print('Classifier: ' + str(outcomes_categories[npa2[i][j]]) + ', with accuracy: ' + str(current_emotion_outcomes[npa2[i][j]]))
                    current_list.append(str(outcomes_categories[npa2[i][j]]))
            statistics_classifiers.append(current_list)
           
        overall_list.append(statistics_classifiers)
        
    y=np.array([np.array(xi) for xi in overall_list])       
    
    savefile = os.path.join(current_path + '/results/', t2 + '_predictions')
    np.save(savefile, y)
    
    print('-----------------------')
    print('end')
    

for myfile2 in glob.glob("weka*category_rates.npy"):
    myfile = os.path.join(current_path + '/results/', myfile2)
    conf_matrix = np.load(myfile)
    num_rows, num_cols = conf_matrix.shape        
    
    view=''
    for i in range(0, num_rows):
        view += ''

    npa = np.sort(conf_matrix,axis=1) 
    npa2 = np.argsort(conf_matrix, axis=1) 
    
    t = myfile.split('weka_predictions_',1)[1]
    t2 = t.split('_all',1)[0]
    print(t2)
    
    print(npa)
    
    overall_list2 = []
    
    for i in range(0, len(conf_matrix)):
        current_emotion_outcomes = conf_matrix[i]
        
        print(np.amin(current_emotion_outcomes))
        print(np.amax(current_emotion_outcomes))
        print(np.std(current_emotion_outcomes))
        print(np.var(current_emotion_outcomes))
    
        statistics = []
        
        
        statistics.append(np.percentile(current_emotion_outcomes, 90))
        statistics.append(np.percentile(current_emotion_outcomes, 75))
        statistics.append(np.median(current_emotion_outcomes))
        statistics.append(np.mean(current_emotion_outcomes))
        
    
        statistics_classifiers = []
        
        for h in range(0, len(statistics)):    
            print('\nClassifiers for statistics[' + str(h) + ']')
            current_list = []
            for j in reversed(range(0, len(current_emotion_outcomes))):
                if current_emotion_outcomes[npa2[i][j]] > statistics[h]:
                    print('Classifier: ' + str(outcomes_categories[npa2[i][j]]) + ', with accuracy: ' + str(current_emotion_outcomes[npa2[i][j]]))
                    current_list.append(str(outcomes_categories[npa2[i][j]]))
            statistics_classifiers.append(current_list)
           
        overall_list2.append(statistics_classifiers)
        
    y=np.array([np.array(xi) for xi in overall_list2])       
    
    savefile = os.path.join(current_path + '/results/', t2 + '_category_rates')
    np.save(savefile, y)
    
    print('-----------------------')
    print('end')