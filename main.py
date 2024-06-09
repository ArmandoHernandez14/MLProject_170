import numpy as np

def Nearest_Neighbor(trainX, trainY, testX, testY):
    m,n = trainX.shape
    nearest_neighbor_index = (np.square(trainX - (np.ones((m, 1)) @ testX)).sum(axis=1)).argmin()
    return trainY[nearest_neighbor_index][0] == testY[0][0]

def Evaluation_Function(x):


    if len(x) == 0:
        column = data[:,0]
        return max(np.sum(column == 1), np.sum(column == 2))/len(column)
    
    feature_data = data[:,list(x)]
    label_data = data[:,0:1]
    m,n = feature_data.shape
    mean = np.ones((m,1)) @ feature_data.mean(axis=0).reshape((1,n))
    stddev = np.ones((m,1)) @ feature_data.std(axis=0).reshape((1,n))
    feature_data = (feature_data - mean)/stddev
    divider = 0
    numberCorrect = 0

    while(divider < m):
        trainX = np.delete(feature_data, divider, axis=0)
        trainY = np.delete(label_data, divider, axis=0)
        testX = feature_data[divider:divider+1,:]
        testY = label_data[divider:divider+1,:]
        numberCorrect = numberCorrect + Nearest_Neighbor(trainX, trainY, testX, testY)
        divider = divider + 1

    return numberCorrect/m
    

def Do_Forward_Selection():
    features_being_used = set({})
    features_not_yet_used = set(range(1, number_of_features+1))
    current_best_accuracy = Evaluation_Function(features_being_used)
    print('Using no features and \"random\" evaluation, I get an accuracy of ', end='')
    print(f"{current_best_accuracy:.1%}\n")
    print('Beginning search\n')
    continue_running = True

    while len(features_not_yet_used) > 0 and continue_running:
        best_feature_index = -1

        for i in features_not_yet_used:   
            candidate_set = features_being_used | {i}
            newest_accuracy = Evaluation_Function(candidate_set)
            print('\tUsing feature(s)', candidate_set, 'accuracy is ', end='')
            print(f"{newest_accuracy:.1%}\n")

            if newest_accuracy >= current_best_accuracy:
                current_best_accuracy = newest_accuracy
                best_feature_index = i
        
        if best_feature_index == -1:
            print('(Warning, Accuracy has decreased!)')
            print('Finished search!! The best feature subset is, ', features_being_used, 'which has an accuracy of ', end='')
            print(f"{current_best_accuracy:.1%}\n")
            continue_running = False

        else:
            features_being_used.update({best_feature_index})
            features_not_yet_used.remove(best_feature_index)
            print('Feature set', features_being_used, 'was best, accuracy is ', end='')
            print(f"{current_best_accuracy:.1%}\n")


def Do_Backward_Selection():
    features_being_used = set(range(1,number_of_features+1))
    current_best_accuracy = Evaluation_Function(features_being_used)
    print('Using all features and \"random\" evaluation, I get an accuracy of ', end='')
    print(f"{current_best_accuracy:.1%}\n")
    print('Beginning search\n')
    continue_running = True

    while len(features_being_used) > 0 and continue_running:
        best_feature_index = -1

        for i in features_being_used:  
            candidate_set = features_being_used.difference({i})
            newest_accuracy = Evaluation_Function(candidate_set)
            print('\tUsing feature(s)', candidate_set, 'accuracy is ', end='')
            print(f"{newest_accuracy:.1%}\n")

            if newest_accuracy > current_best_accuracy:
                current_best_accuracy = newest_accuracy
                best_feature_index = i
        
        if best_feature_index == -1:
            print('(Warning, Accuracy has decreased!)')
            print('Finished search!! The best feature subset is, ', features_being_used, 'which has an accuracy of ', end='')
            print(f"{current_best_accuracy:.1%}\n")
            continue_running = False

        else:
            features_being_used.difference_update({best_feature_index})
            print('Feature set', features_being_used, 'was best, accuracy is ', end='')
            print(f"{current_best_accuracy:.1%}\n")


#data = np.loadtxt("small-test-dataset.txt") #Load data into matrix
#data = np.loadtxt("large-test-dataset.txt") #Load data into matrix
#data = np.loadtxt("CS170_Spring_2024_Small_data__31.txt") #Load data into matrix
data = np.loadtxt("CS170_Spring_2024_Large_data__31.txt") #Load data into matrix


# to use the data set, uncomment the one that you want to use. 

print('Welcome to Ryan Noghani\'s Feature Selection Algorithm.\n')
print("Please enter total number of features: ", end='')
number_of_features = int(input())
print('\nType the number of the algorithm you want to run.\n\n')
print('\tForward Selection')
print('\tBackward Elimination')
print('\tBertie\'s Special Algorithm\n')
choice = input()

num_instances,num_colums  = data.shape
num_features = num_colums -1
print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
if choice == '1':
    Do_Forward_Selection()

elif choice == '2':
    Do_Backward_Selection()

else:
    print(Evaluation_Function(set({1,15,27})))      