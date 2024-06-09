import numpy as np
from itertools import chain, combinations
# Define the nearest neighbor function to find the closest training sample to the test sample
def Nearest_Neighbor(trainX, trainY, testX, testY):
    m, n = trainX.shape  # Get the number of training samples and features
    # Calculate the Euclidean distance between the test sample and each training sample
    nearest_neighbor_index = (np.square(trainX - (np.ones((m, 1)) @ testX)).sum(axis=1)).argmin()
    # Return True if the predicted label matches the actual label, False otherwise
    return trainY[nearest_neighbor_index][0] == testY[0][0]

# Define the evaluation function to calculate the accuracy of a given feature subset
def Evaluation_Function(x):
    if len(x) == 0:
        column = data[:, 0]  # Get the labels from the data
        # Return the accuracy of the majority class prediction
        return max(np.sum(column == 1), np.sum(column == 2)) / len(column)
    
    feature_data = data[:, list(x)]  # Extract the features specified by the subset x
    label_data = data[:, 0:1]  # Extract the labels
    m, n = feature_data.shape  # Get the number of samples and features
    # Normalize the feature data
    mean = np.ones((m, 1)) @ feature_data.mean(axis=0).reshape((1, n))
    stddev = np.ones((m, 1)) @ feature_data.std(axis=0).reshape((1, n))
    feature_data = (feature_data - mean) / stddev
    divider = 0
    numberCorrect = 0

    # Perform leave-one-out cross-validation
    while divider < m:
        trainX = np.delete(feature_data, divider, axis=0)  # Training data excluding the current sample
        trainY = np.delete(label_data, divider, axis=0)  # Training labels excluding the current sample
        testX = feature_data[divider:divider + 1, :]  # Current test sample
        testY = label_data[divider:divider + 1, :]  # Current test label
        numberCorrect += Nearest_Neighbor(trainX, trainY, testX, testY)  # Increment the correct count if the prediction is correct
        divider += 1

    # Return the accuracy of the model
    return numberCorrect / m

# Define the forward selection algorithm to find the best subset of features
def Do_Forward_Selection():
    features_being_used = set({})  # Initialize the set of features being used
    features_not_yet_used = set(range(1, number_of_features + 1))  # Initialize the set of features not yet used
    current_best_accuracy = Evaluation_Function(features_being_used)  # Evaluate the initial accuracy with no features
    print('Using no features and "random" evaluation, I get an accuracy of ', end='')
    print(f"{current_best_accuracy:.1%}\n")
    print('Beginning search\n')
    continue_running = True

    # Loop until there are no more features to try or the search is stopped
    while len(features_not_yet_used) > 0 and continue_running:
        best_feature_index = -1  # Initialize the best feature index

        # Evaluate each feature not yet used
        for i in features_not_yet_used:
            candidate_set = features_being_used | {i}  # Create a candidate set by adding the feature to the current set
            newest_accuracy = Evaluation_Function(candidate_set)  # Evaluate the accuracy of the candidate set
            print('\tUsing feature(s)', candidate_set, 'accuracy is ', end='')
            print(f"{newest_accuracy:.1%}\n")

            # Update the best feature if the accuracy improves
            if newest_accuracy >= current_best_accuracy:
                current_best_accuracy = newest_accuracy
                best_feature_index = i
        
        # If no feature improves the accuracy, stop the search
        if best_feature_index == -1:
            print('(Warning, Accuracy has decreased!)')
            print('Finished search!! The best feature subset is, ', features_being_used, 'which has an accuracy of ', end='')
            print(f"{current_best_accuracy:.1%}\n")
            continue_running = False

        # Otherwise, add the best feature to the set of features being used
        else:
            features_being_used.update({best_feature_index})
            features_not_yet_used.remove(best_feature_index)
            print('Feature set', features_being_used, 'was best, accuracy is ', end='')
            print(f"{current_best_accuracy:.1%}\n")

# Define the backward selection algorithm to find the best subset of features
def Do_Backward_Selection():
    features_being_used = set(range(1, number_of_features + 1))  # Initialize the set of features being used
    current_best_accuracy = Evaluation_Function(features_being_used)  # Evaluate the initial accuracy with all features
    print('Using all features and "random" evaluation, I get an accuracy of ', end='')
    print(f"{current_best_accuracy:.1%}\n")
    print('Beginning search\n')
    continue_running = True

    # Loop until there are no more features to try or the search is stopped
    while len(features_being_used) > 0 and continue_running:
        best_feature_index = -1  # Initialize the best feature index

        # Evaluate each feature being used
        for i in features_being_used:
            candidate_set = features_being_used.difference({i})  # Create a candidate set by removing the feature from the current set
            newest_accuracy = Evaluation_Function(candidate_set)  # Evaluate the accuracy of the candidate set
            print('\tUsing feature(s)', candidate_set, 'accuracy is ', end='')
            print(f"{newest_accuracy:.1%}\n")

            # Update the best feature if the accuracy improves
            if newest_accuracy > current_best_accuracy:
                current_best_accuracy = newest_accuracy
                best_feature_index = i
        
        # If no feature improves the accuracy, stop the search
        if best_feature_index == -1:
            print('(Warning, Accuracy has decreased!)')
            print('Finished search!! The best feature subset is, ', features_being_used, 'which has an accuracy of ', end='')
            print(f"{current_best_accuracy:.1%}\n")
            continue_running = False

        # Otherwise, remove the best feature from the set of features being used
        else:
            features_being_used.difference_update({best_feature_index})
            print('Feature set', features_being_used, 'was best, accuracy is ', end='')
            print(f"{current_best_accuracy:.1%}\n")

#data = np.loadtxt("small-test-dataset.txt") #Load data into matrix
data = np.loadtxt("large-test-dataset.txt") #Load data into matrix
#data = np.loadtxt("CS170_Spring_2024_Small_data__31.txt") #Load data into matrix
#data = np.loadtxt("CS170_Spring_2024_Large_data__31.txt") #Load data into matrix

# to use the data set, uncomment the one that you want to use. 

print('Welcome to Ryan, Armando, Michael, Gabriel, and Kirtana Group 31 Feature Selection Algorithm.\n')
print("Please enter total number of features: ", end='')
number_of_features = int(input())
print('\nType the number of the algorithm you want to run.\n\n')
print('\tForward Selection')
print('\tBackward Elimination')
print('\tGroup31\'s Special Algorithm\n')
choice = input()

num_instances,num_colums  = data.shape
num_features = num_colums -1
print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
if choice == '1':
    Do_Forward_Selection()

elif choice == '2':
    Do_Backward_Selection()

else:   #Custom Algorithm: checks every single possible combinations. Takes very long to execute for large number of features (usually 10 or more features).33
    features_being_used = set(range(1,number_of_features+1))
    x = set(chain.from_iterable(combinations(features_being_used, r) for r in range(len(features_being_used)+1)))
    max_accuracy = 0.0
    best_set = set({})
    for i in x:
        candidate = Evaluation_Function(set(i))
        if (candidate > max_accuracy):
            max_accuracy = candidate
            best_set = i
    print('Finished search!! The best feature subset is', best_set, 'which has an accuracy of ', end='')
    print(f"{max_accuracy:.1%}\n")