from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def load_mnist_data():
    # Loading MNIST Datas
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Casting labels to int
    y = y.astype(int)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    #Step 1 completed
    return train_X, train_y, test_X, test_y

def filter_digits(train_X, train_y, test_X, test_y):
    # We said that we are interested only in 8 and 9 digits
    train_mask = (train_y == 8) | (train_y == 9)
    test_mask = (test_y == 8) | (test_y == 9)

    train_X_filtered, train_y_filtered = train_X[train_mask], train_y[train_mask]
    test_X_filtered, test_y_filtered = test_X[test_mask], test_y[test_mask]

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered



def perform_pca(train_X, test_X, train_y, test_y, n_components):
    # Crete an instance of PCA and fit it to the training data
    pca = PCA(n_components)
    pca.fit(train_X)
    X_train_trs = pca.transform(train_X) # it then transforms the X training set
    X_test_trs = pca.transform(test_X)   # and also the X test set
    #Plot train set with labels and colors
    plt.rcParams.update({'font.size': 18, 'font.family': 'sans serif'})

    fig = plt.figure(figsize = (14,8)); 
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap('Blues')

    normalize = plt.Normalize(vmin=min(test_y), vmax=max(test_y))
    scatter = ax.scatter(X_test_trs[:,0],X_test_trs[:,1], s = test_y * 30, c= test_y, cmap = cmap, norm=normalize)
    ax.set(xlabel = '1st Principal Component (on test set)', ylabel= '2nd Principal Component (on test set)', \
       title =  'PCA with 2 components \n Test target values indicated with color and size')

    cbar = fig.colorbar(scatter, ax=ax)
    plt.show()

    return X_train_trs, X_test_trs



def perform_pca_kernel(train_X, test_X, train_y, test_y, n_components):
    pca_kernel = KernelPCA(n_components, kernel='rbf', fit_inverse_transform=True)
    X_train_trs = pca_kernel.fit_transform(train_X)
    X_test_trs = pca_kernel.transform(test_X)   # and also the X test set
    #Plot train set with labels and colors
    plt.rcParams.update({'font.size': 18, 'font.family': 'sans serif'})

    fig = plt.figure(figsize = (14,8)); 
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap('Blues')

    normalize = plt.Normalize(vmin=min(test_y), vmax=max(test_y))
    scatter = ax.scatter(X_test_trs[:,0],X_test_trs[:,1], s = test_y * 30, c= test_y, cmap = cmap, norm=normalize)
    ax.set(xlabel = '1st Principal Component (on test set)', ylabel= '2nd Principal Component (on test set)', \
       title =  'PCA with 2 components \n Test target values indicated with color and size')

    cbar = fig.colorbar(scatter, ax=ax)
    plt.show()

    return X_train_trs, X_test_trs

def train_decision_tree(train_X, test_X, train_y, test_y):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train_X, train_y)
    accuracy = clf.score(test_X, test_y)
    print(f"Decision Tree Classifier Accuracy: {accuracy:.4f}")


def main():
    print("Hello, MNIST Analysis")
    train_X, train_y, test_X, test_y = load_mnist_data()
    print("Now We will perform PCA and PCA with kernel and plot scatter distribution in PCA space" \
    "Then we will plot only 8 and 9 digits in PCA space")
    
    print("\n--- Tutti i Digit ---")
    train_X_pca, test_X_pca = perform_pca(train_X, test_X, train_y, test_y,2)
    
    print("Accuratezza Decision Tree (Dati originali - 784 componenti):")
    train_decision_tree(train_X, test_X, train_y, test_y)
    print("Accuratezza Decision Tree (PCA - 2 componenti):")
    train_decision_tree(train_X_pca, test_X_pca, train_y, test_y)
    train_X_kernel, test_X_kernel = perform_pca_kernel(train_X, test_X, train_y, test_y,2)
    print("Accuratezza Decision Tree (PCA Kernel - 2 componenti):")
    train_decision_tree(train_X_kernel, test_X_kernel, train_y, test_y)
    print("Accuratezza Decision Tree (PCA 5 componenti):")
    train_X_pca_5, test_X_pca_5 = perform_pca(train_X, test_X, train_y, test_y,5)
    train_decision_tree(train_X_pca_5, test_X_pca_5, train_y, test_y)
    
    print("\n--- Solo Digit 8 e 9 ---")
    f_train_X, f_train_y, f_test_X, f_test_y = filter_digits(train_X, train_y, test_X, test_y)
    f_train_X_pca, f_test_X_pca = perform_pca(f_train_X, f_test_X, f_train_y, f_test_y,2)
    
    print("Accuratezza Decision Tree 8vs9 (Dati originali - 784 componenti):")
    train_decision_tree(f_train_X, f_test_X, f_train_y, f_test_y)
    print("Accuratezza Decision Tree 8vs9 (PCA - 2 componenti):")
    train_decision_tree(f_train_X_pca, f_test_X_pca, f_train_y, f_test_y)
    train_decision_tree(train_X_pca, test_X_pca, train_y, test_y)
    train_X_kernel, test_X_kernel = perform_pca_kernel(train_X_kernel, test_X_kernel, train_y, test_y,2)
    print("Accuratezza Decision Tree (PCA Kernel - 2 componenti):")
    train_decision_tree(train_X_kernel, test_X_kernel, train_y, test_y)
    print("Accuratezza Decision Tree (PCA 5 componenti):")
    train_X_pca_5, test_X_pca_5 = perform_pca(train_X, test_X, train_y, test_y,5)
    train_decision_tree(train_X_pca_5, test_X_pca_5, train_y, test_y)
    
    print("As we can see we have a significant drop in accuracy when we reduce the dimensionality"
    "from 784 to 2 components, this is expeceted tho.")
    
    print("Now let's calculate covariance matrix of the dataset")
    covariance_matrix = np.cov(train_X, rowvar=False)
    print(covariance_matrix)
    print("Let's find eigenvalues and sort them")
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalues.sort()
    print(eigenvalues)
    
    



if __name__ == '__main__':
    main()