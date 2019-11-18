import os
import scipy.io as sio
import numpy as np
import scipy.stats

def bayes_factor(logw0, logw1):
    """null model log-likelihood should
    be logw0, non-null model is logw1
    """
    c = np.amax(logw0)
    logz0 = c + np.log(np.mean(np.exp(logw0 - c)))
    c = np.amax(logw1)
    logz1 = c + np.log(np.mean(np.exp(logw1 - c)))
    return np.exp(logz1 - logz0)

def proj(X , C):
    P = np.eye(C.shape[0]) - C.dot(np.linalg.pinv(C.T.dot(C))).dot(C.T)
    return P.dot(X)

base_dir = "/storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward"
varbvs_dir = "snp_to_brain_output"
info_file = "brain_snp_covars_meancentered_scaled.mat"
info = sio.loadmat(os.path.join(base_dir, info_file))["I_cols"][0]
cols = [str(i[0]) for i in info]
col_map = {}
prefix = "snp_to_brain_"
data = sio.loadmat(os.path.join(base_dir, info_file))
Z = data["Z"]
I = data["I"]
N, P = I.shape
yI_null_lik = {}
bayes_factors = {}
   
for idx, brain in enumerate(cols):
    file_path = os.path.join(base_dir, varbvs_dir, prefix+brain+".mat")
    if os.path.isfile(file_path):
        new_idx = idx+1  
        yI = I[:, idx]
        yI_cor = proj(yI, Z)
        yI_null_lik[new_idx] = np.sum([np.log(scipy.stats.norm.pdf(i)) for i in yI_cor])
        col_map[new_idx] = file_path
        logw1 = scipy.io.loadmat(col_map[new_idx])["lnZ"].ravel()
        logw0 = np.ones(logw1.shape[0]) * yI_null_lik[new_idx]
        bayes_factors[new_idx] = bayes_factor(logw0, logw1)
  
