from sklearn.datasets import fetch_openml, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
import numpy as np

# Loading procedure
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

# Pre processing procedure
def filter_digits(train_X, train_y, test_X, test_y):
    # We said that we are interested only in 8 and 9 digits
    train_mask = (train_y == 8) | (train_y == 9)
    test_mask = (test_y == 8) | (test_y == 9)

    train_X_filtered, train_y_filtered = train_X[train_mask], train_y[train_mask]
    test_X_filtered, test_y_filtered = test_X[test_mask], test_y[test_mask]

    return train_X_filtered, train_y_filtered, test_X_filtered, test_y_filtered

# PCA procedures
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

def pca_reconstruction(train_X, test_X, components):
    pca = PCA(n_components=components)
    pca.fit(train_X)
    X_train_trs = pca.transform(train_X)
    X_test_trs = pca.transform(test_X)
    X_test_reconstructed = pca.inverse_transform(X_test_trs)
    return X_test_reconstructed

# Evaluation procedure
def verify_numerical_reconstruction(original_data, reconstructed_data, n_components):
    mse = np.mean((original_data - reconstructed_data) ** 2)
    print(f"Mean Squared Error (MSE) per la ricostruzione con {n_components} componenti: {mse:.4f}")
    return mse

# Clustering procedure
def artificial_and_clustered_data():
    X, y = make_blobs(n_samples=10, centers=3, n_features=2,random_state=0)
    k_means_result = k_means(X, n_clusters=3, random_state=0)
    return X, y, k_means_result

# Plotting precedure
def plot_clusters(X, y, k_means_result):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k', s=100)
    plt.scatter(k_means_result[0][:, 0], k_means_result[0][:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def perform_elbow_method(X, max_k=10):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)

    for k in K:
        centroids, labels, inertia = k_means(X, n_clusters=k, random_state=42)

        # La distortion è la media delle distanze al quadrato dai centroidi
        # X.shape[0] è corretto: restituisce il numero di righe (cioè di campioni) in X.
        distortions.append(sum(np.min(cdist(X, centroids, 'euclidean'), axis=1)**2) / X.shape[0])

        inertias.append(inertia)
    
        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]
    print("Distortion values:")
    for key, val in mapping1.items():
        print(f'{key} : {val}')

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

# Training procedures
def train_decision_tree(train_X, test_X, train_y, test_y):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train_X, train_y)
    accuracy = clf.score(test_X, test_y)
    print(f"Decision Tree Classifier Accuracy: {accuracy:.4f}")

def train_svm(train_X, test_X, train_y, test_y):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(train_X, train_y)
    accuracy = svm.score(test_X, test_y)
    print(f"SVM Classifier Accuracy: {accuracy:.4f}")

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
    # We are commenting out kernel PCA because it will take a lot of time to compute.
    #train_X_kernel, test_X_kernel = perform_pca_kernel(train_X, test_X, train_y, test_y,2)
    #print("Accuratezza Decision Tree (PCA Kernel - 2 componenti):")
    #train_decision_tree(train_X_kernel, test_X_kernel, train_y, test_y)
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
    #Same motivation as before.
    #train_X_kernel, test_X_kernel = perform_pca_kernel(train_X, test_X, train_y, test_y,2)
    #print("Accuratezza Decision Tree (PCA Kernel - 2 componenti):")
    #train_decision_tree(train_X_kernel, test_X_kernel, train_y, test_y)
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
    print("We have to find out the number of components to retain 80% of the variance")
    #Compuing total variance
    total_variance = np.sum(eigenvalues)
    # Computing the threshold for 80% of variance
    variance_threshold = 0.8 * total_variance
    # Computing the cumulative variance explained by the sorted eigenvalues
    cumulative_variance = np.cumsum(eigenvalues)
    # Computing the number of components
    n_components_80 = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Number of components to retain 80% of the variance: {n_components_80}")

    print("Now let's perform PCA with this number of component")
    train_X_pca, test_X_pca = perform_pca(train_X, test_X, train_y, test_y,n_components_80)
    
    print("We are trying to reconstruct an image from 2 component PCA")
    X_test_reconstructed = pca_reconstruction(train_X, test_X, 2)
    verify_numerical_reconstruction(test_X, X_test_reconstructed, 2)
    
    print("We are trying to reconstruct an image from 5 component PCA")
    X_test_reconstructed = pca_reconstruction(train_X, test_X, 5)
    verify_numerical_reconstruction(test_X, X_test_reconstructed, 5)

    print(f"We are trying to reconstruct an image from {n_components_80} component PCA")
    X_test_reconstructed_80 = pca_reconstruction(train_X, test_X, n_components_80)
    verify_numerical_reconstruction(test_X, X_test_reconstructed_80, n_components_80)
    # By looking at the MSE we see that 2 component are not enough to reconstruct
    # the original data, neither 5 components. But using the same number of component
    # that we need to retain 80% of the variance get us able to reconstruct the original data
    # with 0 error.

    print("Now we will train SVM model and see the accuacy")
    # I'm commenting this because using a kernel SVM on the 784x784 will take a lot of time
    #print("Accuratezza SVM (Dati originali - 784 componenti):")
    #train_svm(train_X, test_X, train_y, test_y)
    print("Accuratezza SVM (PCA - 2 componenti):")
    train_svm(train_X_pca, test_X_pca, train_y, test_y)
    print("Accuratezza SVM (PCA 5 componenti):")
    train_svm(train_X_pca_5, test_X_pca_5, train_y, test_y)

    print("Ok, now we have to: 1. Generate synthetic clustered data. \n " \
    "   2. Plot original data (true clusters. \n" \
    "   3. Apply KMeans. \n " \
    "   4. Plot centroids and KMeans result. \n" \
    "   5. Use elbow method and silhouette scores to find a good class number.")
    X,y, k_result = artificial_and_clustered_data()
    plot_clusters(X,y,k_result)
    perform_elbow_method(X, max_k=10)
          


if __name__ == '__main__':
    main()