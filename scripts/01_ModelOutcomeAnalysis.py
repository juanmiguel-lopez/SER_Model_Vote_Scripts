# Importing necessary libraries and modules
import glob, os, numpy as np
import os

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
        if 'predictions' in elem:
            confussion_matrices.append(elem)            
        elif 'output' in elem:
            classifier_outputs.append(elem)

for mydir in confussion_matrices:
    print('\n'+mydir)
    print('---------------------')
    os.chdir(mydir)

    outcomes = []
    classifiers = []
    
    category_rates = []
    classifiers_category_rates = []

    
    for myfile in glob.glob("*.npy"):
        myfile = os.path.join(mydir, myfile)
        conf_matrix = np.load(myfile)
        num_rows, num_cols = conf_matrix.shape        
        
        
        classifiers.append(myfile)
        if (len(outcomes) == 0):
            outcomes = [[] for i in range(0, num_rows)]
            
        if (len(category_rates) == 0):
            category_rates = [[] for i in range(0, num_rows)]
            
        
        for i in range(0, num_cols):
                    
            mysum = np.sum(conf_matrix, axis=1)
            
            
            outcomes[i].append(conf_matrix[i][i]/mysum[i])
            
            
            mysum2 = np.sum(conf_matrix, axis=0)
            if mysum2[i]<>0:
                category_rates[i].append(conf_matrix[i][i]/mysum2[i])
            else:
                category_rates[i].append(0)
            
            

    y=np.array([np.array(xi) for xi in outcomes])       
    savefile = os.path.join(current_path + '/results/', os.path.basename(mydir))
    np.save(savefile, y)
    
    y2=np.array([np.array(xi) for xi in category_rates])       
    savefile2 = os.path.join(current_path + '/results/', os.path.basename(mydir)+'_category_rates')
    np.save(savefile2, y2)
    

y=np.array([np.array(xi) for xi in classifiers])
savefile = os.path.join(current_path + '/results/', 'classifiers.npy')
np.save(savefile, y)