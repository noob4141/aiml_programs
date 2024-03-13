import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

class NaiveBayesClassifier:
    
    def __init__(self, X, y):
        '''
        X and y denote the features and the target labels respectively
        '''
        self.X, self.y = X, y 
        self.N = len(self.X) # Length of the training set
        self.dim = len(self.X[0]) # Dimension of the vector of features
        self.attrs = [[] for _ in range(self.dim)] # Here we'll store the columns of the training set
        self.output_dom = {} # Output classes with the number of occurrences in the training set. In this case we have only 2 classes
        self.data = [] # To store every row [Xi, yi]
        
        for i in range(len(self.X)):
            for j in range(self.dim):
                # if we have never seen this value for this attr before, 
                # then we add it to the attrs array in the corresponding position
                if not self.X[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.X[i][j])
                    
            # if we have never seen this output class before,
            # then we add it to the output_dom and count one occurrence for now
            if not self.y[i] in self.output_dom.keys():
                self.output_dom[self.y[i]] = 1
            # otherwise, we increment the occurrence of this output in the training set by 1
            else:
                self.output_dom[self.y[i]] += 1
            # store the row
            self.data.append([self.X[i], self.y[i]])

    def classify(self, entry):
        solve = None # Final result
        max_arg = -1 # partial maximum

        for y in self.output_dom.keys():
            prob = self.output_dom[y]/self.N # P(y)

            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] # all rows with Xi = xi
                n = len(cases)
                prob *= n/self.N # P *= P(Xi = xi)
                
            # if we have a greater prob for this output than the partial maximum...
            if prob > max_arg:
                max_arg = prob
                solve = y

        return solve

# Load the data
data = pd.read_csv('titanic.csv')

# Convert target values to strings
y = list(map(lambda v: 'yes' if v == 1 else 'no', data['Survived'].values))

# Features
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']].values

# Initialize the classifier
classifier = NaiveBayesClassifier(X, y)

# Predicting for each data point and storing the result
predictions = [classifier.classify(entry) for entry in X]

# Actual labels
actual_labels = data['Survived'].values

# Convert actual labels to strings
actual_labels = list(map(lambda v: 'yes' if v == 1 else 'no', actual_labels))

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predictions)

# Print accuracy
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(actual_labels, predictions))
