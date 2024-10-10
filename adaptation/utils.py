# reference: https://github.com/deeplearning-wisc/knn-ood/blob/1321bfda6348a7eb2297682c11727f6b284f43d6/util/mahalanobis_lib.py#L171

from __future__ import print_function
import torch
import numpy as np
import sklearn.covariance
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

def sample_estimator(features, labels, num_classes=10):
    """
    compute sample mean and precision (inverse of covariance)
    :param features: feature embeddings in numpy array
    :param labels: corresponding labels in numpy array

    return: class_means: list of class mean
            class_precisions: list of precisions
    """

    # estimate class-conditional mean and precision

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    class_means, class_precisions = [], []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]

        # mean
        class_mean = features[idx].mean(0)

        # precision
        X = features[idx] - class_mean
        group_lasso.fit(X) #(X.cpu().numpy())
        # class_precision = torch.from_numpy(group_lasso.precision_).double().cuda()
        class_precision = group_lasso.precision_.astype(np.float32)  # Convert from double precision to np.float32

        class_means.append(class_mean)
        class_precisions.append(class_precision)

    return {'mean': class_means, 'precision': class_precisions}


def compute_mahalanobis(x, mean, precision):
    """
    :param x: dim=(batch_size, n_features)
    :param mean: dim=(batch_size, n_features)
    :param precision: dim=(batch_size, n_features, n_features)
    """
    zero_f = x - mean
    # term_gaussian = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
    term_gaussian = torch.bmm(torch.bmm(zero_f.unsqueeze(1), precision), zero_f.unsqueeze(-1))


    # Extract the diagonal elements to get the squared distances
    mahalanobis_distance_squared = term_gaussian.squeeze(-1).squeeze(1)

    return mahalanobis_distance_squared