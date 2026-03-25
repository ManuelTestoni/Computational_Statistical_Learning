from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Laboratory Goals:
# 1. Load MNIST Dataset and keep only 8 and 9 digits.
# 2. trains a single decision tree and plots it,
# 3. trains a random forest with different numbers of trees
# 4. trains AdaBoost with different numbers of weak learners
# 5. compares test accuracy across the models.

def load_mnist_data():
    # Loading MNIST Datas
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Casting labels to int
    y = y.astype(int)

    # We said that we are interested only in 8 and 9 digits
    filter_mask = (y == 8) | (y == 9)
    X_filtered, y_filtered = X[filter_mask], y[filter_mask]

    train_X, test_X, train_y, test_y = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )
    
    #Step 1 completed
    return train_X, train_y, test_X, test_y

def train_decision_tree(train_X, train_y, test_X, test_y):
    #Create an instance of Decision Tree Classifier and giving hid train datas  
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_X,train_y)
    #Use it to predict tests
    y_pred = clf.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy:", accuracy)

    # Now we can plot the tree using this simple command:
    tree.plot_tree(clf)
    #Step 3 has now been completed.


# In this two functions of rf and ab we will take as parameter one int called
# estimators in order do create a dynamic instance of a classificator with differnet number
# of basic estimator, asked to the user when the program starts.
def train_random_forest(train_X, train_y, test_X, test_y, estimators):
    # Creating an instance of random forest classifier
    rf = RandomForestClassifier(n_estimators=estimators, random_state=42)
    rf.fit(train_X, train_y)
    #Saving predictions and getting model's accuracy
    y_pred = rf.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy:", accuracy)
    # Step 4 has been completed


def train_ada_boos(train_X, train_y, test_X, test_y, estimators):
    # Creating an instance of AdaBoost
    adaboost = AdaBoostClassifier(n_estimators=estimators, random_state=42)
    adaboost.fit(train_X, train_y)
    # Saving predictions and getting model's accuracy
    y_pred = adaboost.predict(test_X)
    accuracy = accuracy_score(y_pred, test_y)
    print("Accuracy:", accuracy)
    # Step 5 has been completed


def main():
    print("Hi, you are now using a ML model comparator!")
    print("We will be usign MNIST with only 8 and 9 digits, so we are simplifying furthermore the dataset.")
    print("DECISION TREE ")
    train_X,train_y,test_X,test_y = load_mnist_data()
    train_decision_tree(train_X,train_y,test_X,test_y)
    print("Before starting random forest training i have to ask you to insert a number of estimator to test" \
    "variability of ensemble method. \n TIP: Try insertin 10 and next run try inserting 50")
    n_estimators = input()
    print("RANDOM FOREST")
    train_random_forest(train_X, train_y, test_X, test_y, n_estimators)
    print("Before starting ada boost training i have to ask you to insert a number of estimator to test" \
    "variability of ensemble method. \n TIP: Try insertin 10 and next run try inserting 50")
    n_estimators = input()
    print("ADA BOOST")
    train_ada_boos(train_X, train_y, test_X, test_y, n_estimators)
    

if __name__ == "__main__":
    main()