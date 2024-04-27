import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean, pdist, cdist, jensenshannon
from scipy.special import rel_entr
from scipy.stats import ks_2samp, kstest, wasserstein_distance
from sklearn import metrics

def probs(sample, bins):
    """
    Computes probabilities
    """
    pesos = np.ones_like(sample)/len(sample)
    probs, b = np.histogram(sample, weights=pesos, bins=bins)
    return probs, b

def kl_divergence(original, samples):
    p_original, _ = probs(original, bins=100)
    divergences = []
    for s in samples:
        p, _ = probs(s, bins=100)
        p[p==0.0] = 1.e-15 # evita divisao por 0
        d = sum(rel_entr(p_original, p))
        divergences.append(d)
    return divergences
    
# def js_divergence(original, samples, col=0):
#     divergences = []
#     p, _ = probs(original, bins=100)
#     for s in samples:
#         print (s.shape)
#         q, _ = probs(s[:, col], 100)
#         js = jensenshannon(p, q)
#         divergences.append(js)
#     return divergences

def js_divergence(original, samples, col=0):
    """
    Computes the Jensen shannon distance
    Params:
        - original: original dataset 1d-array
        - samples: array of arrays with shape (n, 4)
    """
    divergences = []
    p, _ = probs(original, bins=100)
    for s in samples:
        q, _ = probs(s[:, col], 100)
        js = jensenshannon(p, q)
        divergences.append(js)
    return divergences

def cdf(data, bins):
    count, bins_count = np.histogram(data, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf, pdf, bins_count

def ks_teste(original, fakes, bins=50):
    """
    Returns a list of pvalues for each fake sample in fakes.

    ks_2samp p_value > 0.05 the samples comes from the same distribution
    """
    ks_testes = [ks_2samp(original, f) for f in fakes]
    return ks_testes

def normality_test(res):
    """
    Performs normality test using 
    """
    pvalues = [kstest(res[:, i], 'norm', args=(0, np.std(res[:, i])))[1] for i in range(res.shape[-1])]
    return pvalues

def dtw_distance(X, Y, dist='', mean=False):
    shape_X, shape_Y = X.shape, Y.shape
    if len(shape_X)==3:
        X = X.reshape(shape_X[0]*shape_X[1], shape_X[2])
    if len(shape_Y)==3:
        Y = Y.reshape(shape_Y[0]*shape_Y[1], shape_Y[2])
    distances = []
    if dist=='euclidian':
        distfunc = euclidean
    for i in range(X.shape[-1]):
        d, cost_matrix, acc_cost_matrix, path = dtw(X[:, i].reshape(-1,1), Y[:, i].reshape(-1,1), distfunc)
        distances.append(d)
    if mean:
        return np.mean(distances)
    return distances

def w_distance(X, Y):
    """
    TODO: Compute distance considering the multivariate scenario: https://github.com/PythonOT/POT/issues/182
    """
    shape_X, shape_Y = X.shape, Y.shape
    if len(shape_X)==3:
        X = X.reshape(shape_X[0]*shape_X[1], shape_X[2])
    if len(shape_Y)==3:
        Y = Y.reshape(shape_Y[0]*shape_Y[1], shape_Y[2])
    distances = []
    for i in range(X.shape[-1]):
        r = X[:, i]
        f = Y[:, i]
        pr, _ = probs(r, 50)
        pf, _ = probs(f, 50)
        bins = np.arange(len(pr))
        wd = wasserstein_distance(bins, bins, pr, pf)
        distances.append(wd) 
    return distances

def mmd(X, Y, sigma=1.0):
    """
    Implements the Maximum Mean Discrepancy. Adapted from https://torchdrift.org/notebooks/note_on_mmd.html

    """
    X = np.array(X)
    n, d = X.shape
    m, d2 = Y.shape
    assert (d == d2)
    XY = np.concatenate([X, Y], axis=0)
    dists = cdist(XY, XY)
    k = np.exp((-1/(2*sigma**2)) * dists**2) + np.eye(n+m) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd
