from sklearn.datasets import make_blobs
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from ASKF import BinaryASKFClassifier, ASKFKernels, VectorizedASKFClassifier

#
# binary classification of blob dataset on precomputed kernels
#

X, y = make_blobs(centers=2, n_samples=250, n_features=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

linear_kernel_train = X_train @ X_train.T
poly_kernel_train = (X_train @ X_train.T + 1) ** 2

linear_kernel_test = X_test @ X_train.T
poly_kernel_test = (X_test @ X_train.T + 1) ** 2

# Kernel list needs to be wrapped in ASKFKernels(.)
K_train = ASKFKernels([linear_kernel_train, poly_kernel_train])
K_test = ASKFKernels([linear_kernel_test, poly_kernel_test])

parameters = {
    "c": [1, 10]
}

# other variations from the thesis:
# - canonical (original)
# - canonical faster (reformulated regularization)
clf = GridSearchCV(BinaryASKFClassifier(variation="minmax"), parameters).fit(K_train, y_train)

print("test score", clf.score(K_test,y_test))
