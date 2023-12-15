import glob, os, numpy as np
import sys
import stat

config_lines = []
for line in open(sys.argv[1]):
    li=line.strip()
    if not li.startswith("#"):
        config_lines.append(line.rstrip())
config_lines[:] = [x for x in config_lines if x]

current_path = config_lines[1] + config_lines[0] + "/"

path = os.path.dirname(current_path)

#All subfolders
list_subfolders_with_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

#All subfolders with models and data
list_subfolders_with_paths = [mydir for mydir in list_subfolders_with_paths if mydir.endswith('.arff')==True]

#All subfolders matrices alone
confussion_matrices = []
classifier_outputs = []
for mydir in list_subfolders_with_paths:
    matrices_folders = [x[0] for x in os.walk(mydir)]
    for elem in matrices_folders:
        if 'predictions' in elem:
            confussion_matrices.append(elem)
            #all_matrices.extend(matrices_folders)   
        elif 'output' in elem:
            classifier_outputs.append(elem)
      
#Select classifiers por each dataset based on confussion matrix outcomes
for mydir in confussion_matrices:
    #data_folder = os.walk(mydir) #os.path.join("source_data", "text_files")
    print('\n'+mydir)
    print('---------------------')
    os.chdir(mydir)

    outcomes = []
    classifiers = []
    
    #Confussion matrices: Accuracy
    for myfile in glob.glob("*.npy"):
        conf_matrix = np.load(myfile, mmap_mode='r')
        num_rows, num_cols = conf_matrix.shape        
        
        classifiers.append(myfile)
        if (len(outcomes) == 0):
            outcomes = [[] for i in range(0, num_rows)]
        
        for i in range(0, num_cols):
            #axis=1, calculate sum amongst rows, elements per emotion
            mysum = np.sum(conf_matrix, axis=1)
            outcomes[i].append(conf_matrix[i][i]/mysum[i])

    y = np.array([np.array(xi) for xi in outcomes])
       
    
    savefile = os.path.basename(mydir)
    np.save(savefile, y)

y = np.array([np.array(xi) for xi in classifiers])

savefile = 'classifiers.npy'
np.save(savefile, y)