import numpy as np

def Nearest_Neighbor(trainX, trainY, testX, testY):
    """
    Function to find the nearest neighbor for a test instance
    based on the training data and check if its label matches the test label.

    Parameters:
    trainX (ndarray): Training data features
    trainY (ndarray): Training data labels
    testX (ndarray): Test data feature (single instance)
    testY (ndarray): Test data label (single instance)

    Returns:
    bool: True if the nearest neighbor's label matches the test label, False otherwise
    """
    m, n = trainX.shape
    nearest_neighbor_index = (np.square(trainX - (np.ones((m, 1)) @ testX)).sum(axis=1)).argmin()
    return trainY[nearest_neighbor_index][0] == testY[0][0]

def Evaluation_Function(x):
    """
    Function to evaluate the accuracy of a feature subset using leave-one-out cross-validation.

    Parameters:
    x (set): Set of feature indices to be used for evaluation

    Returns:
    float: Accuracy of the feature subset
    """
    if len(x) == 0:
        column = data[:, 0]
        return max(np.sum(column == 1), np.sum(column == 2)) / len(column)
    
    feature_data = data[:, list(x)]
    label_data = data[:, 0:1]
    m, n = feature_data.shape

    # Normalize feature data
    mean = np.ones((m, 1)) @ feature_data.mean(axis=0).reshape((1, n))
    stddev = np.ones((m, 1)) @ feature_data.std(axis=0).reshape((1, n))
    feature_data = (feature_data - mean) / stddev

    divider = 0
    numberCorrect = 0

    # Leave-one-out cross-validation
    while divider < m:
        trainX = np.delete(feature_data, divider, axis=0)
        trainY = np.delete(label_data, divider, axis=0)
        testX = feature_data[divider:divider+1, :]
        testY = label_data[divider:divider+1, :]
        numberCorrect += Nearest_Neighbor(trainX, trainY, testX, testY)
        divider += 1

    return numberCorrect / m

def Do_Forward_Selection():
    """
    Function to perform forward feature selection and print the results.
    """
    features_being_used = set({})
    features_not_yet_used = set(range(1, number_of_features + 1))
    current_best_accuracy = Evaluation_Function(features_being_used)

    print('Using no features and "random" evaluation, I get an accuracy of ', end='')
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
    """
    Function to perform backward feature elimination and print the results.
    """
    features_being_used = set(range(1, number_of_features + 1))
    current_best_accuracy = Evaluation_Function(features_being_used)

    print('Using all features and "random" evaluation, I get an accuracy of ', end='')
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

# Load data into matrix
data = np.loadtxt("large-test-dataset.txt")

print('Welcome to Ryan Noghani\'s Feature Selection Algorithm.\n')
print("Please enter total number of features: ", end='')
number_of_features = int(input())
print('\nType the number of the algorithm you want to run.\n\n')
print('\t1. Forward Selection')
print('\t2. Backward Elimination')
print('\t3. Bertie\'s Special Algorithm\n')

choice = input()

if choice == '1':
    Do_Forward_Selection()
elif choice == '2':
    Do_Backward_Selection()
else:
    print(Evaluation_Function(set({1, 15, 27})))
