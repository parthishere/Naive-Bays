import random
import math
import pandas as pd


data = pd.read_csv('netflix_titles_enriched.csv')
data = data.drop(columns=['rottentomatoes_info', 'rottentomatoes_cast', 'rottentomatoes_audience_#reviews', 'rottentomatoes_audience_review',
                 'rottentomatoes_tomatometer_score', 'rottentomatoes_critics_#reviews', 'rottentomatoes_critic_review'])
data = data.drop(columns=['cast'])
data = data.drop(columns=data.columns[0])
data = data.dropna()

print(data.info())
print(data.head())

data['duration'] = data['duration'].map(
    lambda x: x.lstrip().rstrip(' min Seasons'))

dataM = data[data['type'] == "Movie"]
dataT = data[data['type'] == "TV Show"]

dataM.loc['duration'] = pd.to_numeric(dataM['duration'])

tsize = int(0.8 * dataM.shape[0])
vsize = dataM.shape[0] - tsize

td = dataM.iloc[:tsize, :]
vd = dataM.iloc[tsize:, :]

training_data = td.values.tolist()
testing_data = vd.values.tolist()

header = ["Cast", "Type", "Title", "Country", "Date Added", "Release Year",
          "Description", "Director", "Duration", "Genre", "Rating", "Rotten Tomatoes Rating"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

    return probs


class NaiveBays:

    # the categorical class names are changed to numberic data
    # eg: yes and no encoded to 1 and 0
    def encode_class(mydata):
        classes = []
        for i in range(len(mydata)):
            if mydata[i][-1] not in classes:
                classes.append(mydata[i][-1])
        for i in range(len(classes)):
            for j in range(len(mydata)):
                if mydata[j][-1] == classes[i]:
                    mydata[j][-1] = i
        return mydata

    # Splitting the data

    def splitting(mydata, ratio):
        train_num = int(len(mydata) * ratio)
        train = []
        # initially testset will have all the dataset
        test = list(mydata)
        while len(train) < train_num:
            # index generated randomly from range 0
            # to length of testset
            index = random.randrange(len(test))
            # from testset, pop data rows and put it in train
            train.append(test.pop(index))
        return train, test

    # Group the data rows under each class yes or
    # no in dictionary eg: dict[yes] and dict[no]

    def groupUnderClass(mydata):
        dict = {}
        for i in range(len(mydata)):
            if (mydata[i][-1] not in dict):
                dict[mydata[i][-1]] = []
            dict[mydata[i][-1]].append(mydata[i])
        return dict

    # Calculating Mean

    def mean(numbers):
        return sum(numbers) / float(len(numbers))

    # Calculating Standard Deviation
    def std_dev(numbers):
        avg = mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / \
            float(len(numbers) - 1)
        return math.sqrt(variance)

    def MeanAndStdDev(mydata):
        info = [(mean(attribute), std_dev(attribute))
                for attribute in zip(*mydata)]
        # eg: list = [ [a, b, c], [m, n, o], [x, y, z]]
        # here mean of 1st attribute =(a + m+x), mean of 2nd attribute = (b + n+y)/3
        # delete summaries of last class
        del info[-1]
        return info

    # find Mean and Standard Deviation under each class
    def MeanAndStdDevForClass(mydata):
        info = {}
        dict = groupUnderClass(mydata)
        for classValue, instances in dict.items():
            info[classValue] = MeanAndStdDev(instances)
        return info

    # Calculate Gaussian Probability Density Function

    def calculateGaussianProbability(x, mean, stdev):
        expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo

    # Calculate Class Probabilities

    def calculateClassProbabilities(info, test):
        probabilities = {}
        for classValue, classSummaries in info.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, std_dev = classSummaries[i]
                x = test[i]
                probabilities[classValue] *= calculateGaussianProbability(
                    x, mean, std_dev)
        return probabilities

    # Make prediction - highest probability is the prediction

    def predict(info, test):
        probabilities = calculateClassProbabilities(info, test)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    # returns predictions for a set of examples

    def getPredictions(info, test):
        predictions = []
        for i in range(len(test)):
            result = predict(info, test[i])
            predictions.append(result)
        return predictions

    # Accuracy score
    def accuracy_rate(test, predictions):
        correct = 0
        for i in range(len(test)):
            if test[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(test))) * 100.0


# driver code

# add the data path in your system
# mydata = list(mydata)
# mydata = encode_class(mydata)
# for i in range(len(mydata)):
#     mydata[i] = [float(x) for x in mydata[i]]


# # split ratio = 0.7
# # 70% of data is training data and 30% is test data used for testing
# ratio = 0.7
# train_data, test_data = splitting(mydata, ratio)
# print('Total number of examples are: ', len(mydata))
# print('Out of these, training examples are: ', len(train_data))
# print("Test examples are: ", len(test_data))

# # prepare model
# info = MeanAndStdDevForClass(train_data)

# # test model
# predictions = getPredictions(info, test_data)
# accuracy = accuracy_rate(test_data, predictions)
# print("Accuracy of your model is: ", accuracy)


# if __name__ == '__main__':

#     my_tree = build_tree(training_data)

#     print_tree(my_tree)

#     # Evaluate

#     count = 0

#     for row in testing_data:
#         results = print_leaf(classify(row, my_tree))
#         fv = next(iter(results))
#         # print(fv)
#         if abs(fv - row[-1]) <= 15:
#             count += 1
#         print("Actual: %s. Predicted: %s" %
#               (row[-1], results))

# print(count)
print(vsize)
print(tsize+vsize)
print(td.shape[0])
print(vd.shape[0])

#datalist = dataM.values.tolist()

# print(data.dtypes)
# print(tsize)
# print(data.info())
# print(data['duration'])

# Importing library
