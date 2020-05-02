# Bayesian model for joint unmixing and classification of hyperspectral images

This code implements the method described in

Lagrange, A., Fauvel, M., May, S., & Dobigeon, N. (2019). Hierarchical Bayesian image analysis: From low-level modeling to robust supervised learning. Pattern Recognition, 85, 26-36.

For the sake of comparison, this code additionally proposes an implementation of the method described in the following paper

Eches, O., Dobigeon, N., & Tourneret, J. Y. (2011). Enhancing hyperspectral image unmixing with spatial correlations. IEEE Transactions on Geoscience and Remote Sensing, 49(11), 4239-4247.

# Usage
An exemple of how to use the code is provided in the file runExpl.py.

Example call: 

`$ mcmc = modelJointClassifUnmix.BayesJointClassifUnmix(gamma=0.1,beta1=beta1,beta2=beta2,preprocBeta=preprocBeta,onlineEstim=True)`

`$ [A_est,Z_est,classif_est,Q_est,psi_est,sigma2_est,activeCluster] = mcmc.sampleModelJointClassifUnmix(im,M,K,C,labels,confidence,Nmc,Nburn=Nburn,mask=mask,equiprobC=bool(equiprobC),printEnable=printEnable,srand=randseed,nproc=nproc)`

## Class initialization inputs

BayesJointClassifUnmix(gamma, beta1, beta2=0., preprocBeta=None, onlineEstim=True)

- gamma:        hyperparamater of inverse-gamma used as prior for gaussian cluster variance (cf paper)

- beta1:        granularity parameter for MRF used as prior for cluster labels z_p. (Used only in Eches model.)

- beta2:        granularity parameter for MRF used as prior for classif labels c_p. Set equal to beta1 if not specified.

- preprocBeta:  granularity parameter for MRF used as prior for cluster labels z_p in the joint classif/unmixing model. Set equal to 0 after burnin period.

- onlineEstim:  (bool) If False, outputs are all the generated samples ; if True, outputs are directly MMSE and MAP estimators introduced in the paper. When processing big images, it is best to use the latter option since storing all samples necessitate extensive memory usage. Default True.

## I/O for Lagrange et al. method

BayesJointClassifUnmix.sampleModelJointClassifUnmix(Y, M, K, C, labels, confidence, Nmc, Nburn=0, mask=None, equiprobC=True, printEnable=False, initPsi=None, initSigma=None, srand=0, nproc=0)

Inputs:
  - Y:           image (spatial dimension 1 x spatial dimension 2 x spectral dimension=d).
  - M:           endmembers matrix with R spectra (dxR).
  - K:           initial number of clusters.
  - C:           number of classes for classification.
  - labels:      training label map for classification. Value 0 stands for label unknown (spatial dimension 1 x spatial dimension 2).
  - confidence:  confidence in provided training labels. Values should be in interval (0. 1.) with 0. and 1. excluded. (spatial dimension 1 x spatial dimension 2).
  - Nmc:         total number of samples drawn for each variable (including burnin period).
  - Nburn:       number of iteration for burnin period
  - mask:        (bool) boolean mask used to exclude pixels from the estimation (False value to exclude) (spatial dimension 1 x spatial dimension 2). Default all True.
  - equiprobC:   (bool) prior probabilities of classes for unlabeled pixels is equiprobable when True, whereas it is equal to proportion of each labels in training set when False. 
  - printEnable: (bool) print information when True. Default False.
  - initPsi:     (optional) initialization of psi_k.
  - initSigma:   (optional) initialization of sigma_k^2.
  - srand:       random seed.
  - nproc:       number of subprocesses to use for computation (default: no subprocesses)

Ouputs:
  - A_tab:         estimated abundance matrix (or samples if onlineEstim is False).
  - Z_tab:         estimated cluster labels (or samples if onlineEstim is False).
  - C_tab:         estimated classification labels (or samples if onlineEstim is False).
  - Q_tab:         estimated interaction matrix (or samples if onlineEstim is False).
  - psi_tab:       estimated clusters means (or samples if onlineEstim is False).
  - sigma2_tab:    estimated clusters variances (or samples if onlineEstim is False).
  - activeCluster: list of active clusters. Sigma2 and psi of inactive clusters should not be considered by users.

## I/O for Eches et al. method

BayesJointClassifUnmix.sampleModelEches(Y, M, K, Nmc, Nburn=0, mask=None, printEnable=False, initPsi=None, initSigma=None, srand=0, nproc=0)

Inputs:
  - Y:           image (spatial dimension 1 x spatial dimension 2 x spectral dimension=d).
  - M:           endmembers matrix with R spectra (dxR).
  - K:           initial number of clusters.
  - Nmc:         total number of samples drawn for each variable (including burnin period).
  - Nburn:       number of iteration for burnin period
  - mask:        (bool) boolean mask used to exclude pixels from the estimation (False value to exclude) (spatial dimension 1 x spatial dimension 2). Default all True.
  - printEnable: (bool) print information when True. Default False.
  - initPsi:     (optional) initialization of psi_k.
  - initSigma:   (optional) initialization of sigma_k^2.
  - srand:       random seed.
  - nproc:       number of subprocesses to use for computation (default: no subprocesses)

Ouputs:
  - A_tab:         estimated abundance matrix (or samples if onlineEstim is False).
  - Z_tab:         estimated cluster labels (or samples if onlineEstim is False).
  - psi_tab:       estimated clusters means (or samples if onlineEstim is False).
  - sigma2_tab:    estimated clusters variances (or samples if onlineEstim is False).
  - activeCluster: list of active clusters. Sigma2 and psi of inactive clusters should not be considered by users.


# Authors
Author: Adrien Lagrange (ad.lagrange@gmail.com)

Under Apache 2.0 license
