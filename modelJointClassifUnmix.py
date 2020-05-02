# -*- coding: utf-8 -*-

import time
import scipy as sp
import scipy.linalg as splin
from matplotlib import pyplot as plt
import rtnorm as rt
import multiprocessing as mp


def multivariate_normal_wrapper(args):
  return sp.random.multivariate_normal(*args)

def choice_wrapper(args):
  return sp.random.choice(*args)

class BayesJointClassifUnmix(object):
  """
    Hierarchical bayesien model to perform joint classification and unmixing
    gamma:        hyperparamater of inverse-gamma used as prior for gaussian cluster variance (cf paper)
    beta1:        granularity parameter for MRF used as prior for cluster labels z_p. (Used only in Eches model.)
    beta2:        granularity parameter for MRF used as prior for classif labels c_p. Set equal to beta1 if not specified.
    preprocBeta:  granularity parameter for MRF used as prior for cluster labels z_p in the joint classif/unmixing model. Set equal to 0 after burnin period.
    onlineEstim:  (bool) If False, outputs are all the generated samples ; if True, outputs are directly MMSE and MAP estimators introduced in the paper. When processing big images, it is best to use the latter option since storing all samples necessitate extensive memory usage. Default True.
  """
  def __init__(self, gamma, beta1, beta2=0., preprocBeta=None, onlineEstim=True):
    super(BayesJointClassifUnmix, self).__init__()
    self.gamma       = gamma
    self.beta1       = beta1
    self.beta2       = beta2
    self.preprocBeta = preprocBeta
    self.onlineEstim = onlineEstim

  def sampleModelEches(self,Y,M,K,Nmc,Nburn=0,mask=None,printEnable=False,initPsi=None,initSigma=None,srand=0,nproc=0):
    """
      Use MCMC estimation method to infer results given by Eches model.
      Input:
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
        - nproc:       number of subprocesses to use (default: no subprocesses)

      Ouput:
        - A_tab:         estimated abundance matrix (or samples if onlineEstim is False).
        - Z_tab:         estimated cluster labels (or samples if onlineEstim is False).
        - psi_tab:       estimated clusters means (or samples if onlineEstim is False).
        - sigma2_tab:    estimated clusters variances (or samples if onlineEstim is False).
        - activeCluster: list of active clusters. Sigma2 and psi of inactive clusters should not be considered by users.

      NB: if the dataset is composed of several images, the easiest way to use this code is to concatenate images adding a border between images and then exclude the borders using mask parameters. Indeed, spatial dependency is ineffective throw pixels included in mask.
    """

    # init random number generator seed
    sp.random.seed(srand)

    # Init parameters
    (heightImg, widthImg, L) = Y.shape
    R                        = M.shape[1]
    P                        = heightImg*widthImg
    if mask is None:
      mask = sp.ones((heightImg, widthImg),dtype=bool)

    # Reshape data
    Y      = Y.reshape((P,L)).T

    # MC initialization
    A_est      = sp.ones((R,P),dtype=float)/R
    Z_est      = sp.random.choice(K,P)
    s2_est     = 1.
    if initPsi is None:
      psi_est    = sp.random.dirichlet(sp.ones(R),K).T
    else:
      psi_est    = initPsi
    if initSigma is None:
      sigma2_est = 0.5*sp.ones((R,K))
    else:
      sigma2_est = initSigma
    delta_est  = 1.

    # MC chains
    A_tab,Z_tab,s2_tab,psi_tab,sigma2_tab,delta_tab  = None,None,None,None,None,None
    if self.onlineEstim:
      A_tab      = sp.zeros((R,P))
      Z_tab      = sp.zeros((K,P))
      s2_tab     = 0.
      psi_tab    = sp.zeros((R,K))
      sigma2_tab = sp.zeros((R,K))
      delta_tab  = 0.
    else:
      A_tab      = sp.zeros((Nmc,R,P))
      Z_tab      = sp.zeros((Nmc,P))
      s2_tab     = sp.zeros((Nmc))
      psi_tab    = sp.zeros((Nmc,R,K))
      sigma2_tab = sp.zeros((Nmc,R,K))
      delta_tab  = sp.zeros((Nmc))

    ## making the checkboard
    indexA = sp.arange(heightImg*widthImg).reshape((heightImg,widthImg))

    # black indexes
    tmp1 = indexA[::2,::2]
    tmp2 = indexA[1::2,1::2]
    indA_b = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    # white indexes
    tmp1 = indexA[::2,1::2]
    tmp2 = indexA[1::2,::2]
    indA_w = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    indexZ = sp.arange((heightImg+2)*(widthImg+2)).reshape((heightImg+2,widthImg+2))

    # black indexes
    tmp1 = indexZ[1:-1:2,1:-1:2]
    tmp2 = indexZ[2:-1:2,2:-1:2]
    indZ_b = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    # white indexes
    tmp1 = indexZ[1:-1:2,2:-1:2]
    tmp2 = indexZ[2:-1:2,1:-1:2]
    indZ_w = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    del tmp1, tmp2

    # black neighborhood
    Vb =  sp.vstack(( indZ_b-1, indZ_b+1, indZ_b-(widthImg+2), indZ_b+(widthImg+2) ))
    # white neighborhood
    Vw =  sp.vstack(( indZ_w-1, indZ_w+1, indZ_w-(widthImg+2), indZ_w+(widthImg+2) ))

    proba_b       = sp.empty((K,len(indZ_b)))
    proba_w       = sp.empty((K,len(indZ_w)))
    Z_2d          = sp.zeros((heightImg+2,widthImg+2))-1
    fY            = sp.empty((K,P))
    Lambda        = sp.empty((K,R,R))
    Lambda_inv    = sp.empty((K,R,R))
    mu_param      = sp.empty((K,R))
    activeCluster = list(range(K))
    
    if printEnable:
      plt.ion()
      fig = plt.figure()
      plt.title('Cluster labels')

    ## Sampling
    print( "Log: start sampling..." )
    if nproc == 'all':
      pool=mp.Pool(processes=mp.cpu_count())
    elif nproc>0:
      pool=mp.Pool(nproc)
    start_time = time.time()
    for iterMc in range(Nmc):
      print( "Log: iteration "+str(iterMc+1)+" ("+str(time.time() - start_time)+")" )

      # sampling abondances
      for k in activeCluster:
        Lambda_inv[k,:,:] = sp.dot(M.T,M)/s2_est + sp.diag(1/sigma2_est[:,k])
        Lambda[k,:,:] = splin.pinv(Lambda_inv[k,:,:])
        mu_param[k,:] = (1. /sigma2_est[:,k]) * psi_est[:,k]

      for k in activeCluster:
        indK = Z_est == k
        A_est[:,indK] = sp.random.multivariate_normal(sp.zeros(A_est.shape[0]), Lambda[k,:,:],size=indK.sum()).T
        A_est[:,indK] += sp.dot(Lambda[k,:,:], sp.dot(M.T,Y[:,indK])/s2_est + mu_param[k,:,sp.newaxis])

      # sampling cluster label
      Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))

      for k in activeCluster:
        fY[k,:]  = sp.exp( -0.5*sp.sum(sp.log(sigma2_est[:,k])) - 0.5 * sp.sum( (1. /sigma2_est[:,k,sp.newaxis]) * (A_est - psi_est[:,k,sp.newaxis])**2, axis=0) )

      # update black
      Z_b = Z_2d.ravel()[Vb]
      for k in activeCluster:
        proba_b[k,:] = fY[k,indA_b] * sp.exp(self.beta1*sp.sum(Z_b==k,axis=0))
      proba_b[:,sp.where(sp.sum(proba_b[activeCluster,:],axis=0)<=0)[0]] = 1
      proba_b[activeCluster,:] = proba_b[activeCluster,:]/sp.sum(proba_b[activeCluster,:],axis=0)
      if nproc == 0:
        Z_est[indA_b] = sp.asarray( [sp.random.choice(activeCluster,1,p=proba_b[activeCluster,x]) for x in range(proba_b.shape[1])] ).ravel()
      else:
        Z_est[indA_b] = sp.asarray( pool.map(choice_wrapper, [(activeCluster,1,True,proba_b[activeCluster,x],) for x in range(proba_b.shape[1])]) ).ravel()
      Z_est[sp.logical_not(mask.ravel())] = -1

      # update white
      Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))
      Z_w = Z_2d.ravel()[Vw]
      for k in activeCluster:
        proba_w[k,:] = fY[k,indA_w] * sp.exp(self.beta1*sp.sum(Z_w==k,axis=0))
      proba_w[:,sp.where(sp.sum(proba_w[activeCluster,:],axis=0)<=0)[0]] = 1
      proba_w[activeCluster,:] = proba_w[activeCluster,:]/sp.sum(proba_w[activeCluster,:],axis=0)
      if nproc == 0:
        Z_est[indA_w] = sp.asarray( [sp.random.choice(activeCluster,1,p=proba_w[activeCluster,x]) for x in range(proba_w.shape[1])] ).ravel()
      else:
        Z_est[indA_w] = sp.asarray( pool.map(choice_wrapper, [(activeCluster,1,True,proba_w[activeCluster,x],) for x in range(proba_w.shape[1])]) ).ravel()
      Z_est[sp.logical_not(mask.ravel())] = -1

      # sampling noise variance
      s2_est = 1. / sp.random.gamma(1+P*L/2, 1/(delta_est + 0.5* sp.sum((Y - sp.dot(M,A_est))**2)) )
      
      # sampling cluster mean
      for k in activeCluster:
        ind = Z_est==k
        nk = float(sp.sum(ind))

        if nk!=0:
          indR = sp.random.permutation(R)
          for r in indR[:-1]:
            psi_est[r,k] = rt.rtnorm(0, 1-(sp.sum(psi_est[indR[:-1],k])-psi_est[r,k]), mu=(1/nk)*sp.sum(A_est[r,ind]), sigma=sp.sqrt(sigma2_est[r,k]/nk), size=1)
          psi_est[indR[-1],k] = 1-sp.sum(psi_est[indR[:-1],k])
        else:
          activeCluster.remove(k)

      # sampling cluster variance
      for k in activeCluster:
        ind = Z_est==k
        nk = float(sp.sum(ind))

        param2 = self.gamma + 0.5* sp.sum( (A_est[:,ind] - psi_est[:,k,sp.newaxis])**2, axis=1 )

        sigma2_est[:,k] = 1. / sp.random.gamma(nk/2 +1, 1./param2)
      del param2

      # sampling hyperparameter of noise variance (delta)
      delta_est = 1. / sp.random.gamma(1.,1/s2_est)

      # store sampling
      if self.onlineEstim:
        if iterMc>=Nburn:
          N = float(iterMc - Nburn)
          s2_tab     = (N * s2_tab + s2_est) / (1+N)
          delta_tab  = (N * delta_tab + delta_est) / (1+N)
          A_tab      = (N * A_tab + A_est) / (1+N)
          psi_tab    = (N * psi_tab + psi_est) / (1+N)
          sigma2_tab = (N * sigma2_tab + sigma2_est) / (1+N)
          Z_tab[Z_est,sp.arange(Z_est.shape[0])] += 1
      else:
        A_tab[iterMc,:,:]      = A_est
        Z_tab[iterMc,:]        = Z_est
        s2_tab[iterMc]         = s2_est
        psi_tab[iterMc,:,:]    = psi_est
        sigma2_tab[iterMc,:,:] = sigma2_est
        delta_tab[iterMc]      = delta_est

      if printEnable:
        Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))
        plt.imshow(Z_2d,interpolation='none')
        fig.canvas.draw()

    if nproc != 0:
      pool.close()
      pool.terminate()
    
    if printEnable:
      plt.close()
      plt.ioff()

    if self.onlineEstim:
      Z_tab = sp.argmax(Z_tab,axis=0)
      Z_tab[sp.logical_not(mask).ravel()] = -1

    return A_tab,Z_tab,psi_tab,sigma2_tab,activeCluster

  def sampleModelJointClassifUnmix(self,Y,M,K,C,labels,confidence,Nmc,Nburn=0,mask=None,equiprobC=True,printEnable=False,initPsi=None,initSigma=None,srand=0,nproc=0):
    """
      Use MCMC estimation method to infer results given by model proposed in paper.
      Input:
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

      Ouput:
        - A_tab:         estimated abundance matrix (or samples if onlineEstim is False).
        - Z_tab:         estimated cluster labels (or samples if onlineEstim is False).
        - C_tab:         estimated classification labels (or samples if onlineEstim is False).
        - Q_tab:         estimated interaction matrix (or samples if onlineEstim is False).
        - psi_tab:       estimated clusters means (or samples if onlineEstim is False).
        - sigma2_tab:    estimated clusters variances (or samples if onlineEstim is False).
        - activeCluster: list of active clusters. Sigma2 and psi of inactive clusters should not be considered by users.

      NB: if the dataset is composed of several images, the easiest way to use this code is to concatenate images adding a border between images and then exclude the borders using mask parameters. Indeed, spatial dependency is ineffective throw pixels included in mask.
    """

    # init random number generator seed
    sp.random.seed(srand)

    # Init parameters
    (heightImg, widthImg, L) = Y.shape
    P      = heightImg*widthImg
    R      = M.shape[1]
    if mask is None:
      mask = sp.ones((heightImg, widthImg),dtype=bool)
    if self.preprocBeta != None:
      self.beta1 = self.preprocBeta

    # Reshape data
    Y          = Y.reshape((P,L)).T
    labels     = labels.reshape((P))
    confidence = confidence.reshape((P,1))


    # Separate labeled from non labeled samples and generate confidence
    indUSpl = sp.where(labels == 0)[0]
    indLSpl = sp.where(labels != 0)[0]
    confLabels = sp.ones((P,C))/C
    confLabels[indLSpl] = (1./(C-1)) * ( sp.ones((len(indLSpl),C)) - confidence[indLSpl,:] )
    confLabels[indLSpl,labels[indLSpl]-1] = confidence[indLSpl,0]

    # Get label proportion
    propC = sp.ones((C),dtype=float)
    if equiprobC == False:
      for c in range(C):
        propC[c] = sp.sum(labels[indLSpl] == c+1)
      propC /= sp.sum(propC)
    confLabels[indUSpl,:] = confLabels[indUSpl,:] * propC

    # MC initialization
    A_est          = sp.ones((R,P),dtype=float)/R
    Z_est          = sp.random.choice(K,P)
    Q_est          = sp.ones((K,C),dtype=float)/K#sp.random.dirichlet(sp.ones(K),C).T
    C_est          = sp.random.choice(C,P)
    C_est[indLSpl] = labels[indLSpl]-1
    C_est[sp.logical_not(mask).ravel()] = -1
    s2_est         = 1.
    if initPsi is None:
      psi_est    = sp.random.dirichlet(sp.ones(R),K).T
    else:
      psi_est    = initPsi
    if initSigma is None:
      sigma2_est = 0.5*sp.ones((R,K))
    else:
      sigma2_est = initSigma
    delta_est  = 1.

    # MC chains
    A_tab,Z_tab,Q_tab,C_tab,s2_tab,psi_tab,sigma2_tab,delta_tab  = None,None,None,None,None,None,None,None
    if self.onlineEstim:
      A_tab      = sp.zeros((R,P))
      Z_tab      = sp.zeros((K,P))
      Q_tab      = sp.zeros((K,C))
      C_tab      = sp.zeros((C,P))
      s2_tab     = 0
      psi_tab    = sp.zeros((R,K))
      sigma2_tab = sp.zeros((R,K))
      delta_tab  = 0
    else:
      A_tab      = sp.zeros((Nmc,R,P))
      Z_tab      = sp.zeros((Nmc,P))
      Q_tab      = sp.zeros((Nmc,K,C))
      C_tab      = sp.zeros((Nmc,P))
      s2_tab     = sp.zeros((Nmc))
      psi_tab    = sp.zeros((Nmc,R,K))
      sigma2_tab = sp.zeros((Nmc,R,K))
      delta_tab  = sp.zeros((Nmc))

    ## making the checkboard
    indexA = sp.arange(heightImg*widthImg).reshape((heightImg,widthImg))

    # black indexes
    tmp1 = indexA[::2,::2]
    tmp2 = indexA[1::2,1::2]
    indA_b = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    # white indexes
    tmp1 = indexA[::2,1::2]
    tmp2 = indexA[1::2,::2]
    indA_w = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    indexZ = sp.arange((heightImg+2)*(widthImg+2)).reshape((heightImg+2,widthImg+2))

    # black indexes
    tmp1 = indexZ[1:-1:2,1:-1:2]
    tmp2 = indexZ[2:-1:2,2:-1:2]
    indZ_b = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    # white indexes
    tmp1 = indexZ[1:-1:2,2:-1:2]
    tmp2 = indexZ[2:-1:2,1:-1:2]
    indZ_w = sp.hstack((tmp1.ravel(), tmp2.ravel()))

    del tmp1, tmp2

    # black neighborhood
    Vb =  sp.vstack(( indZ_b-1, indZ_b+1, indZ_b-(widthImg+2), indZ_b+(widthImg+2) ))
    # white neighborhood
    Vw =  sp.vstack(( indZ_w-1, indZ_w+1, indZ_w-(widthImg+2), indZ_w+(widthImg+2) ))

    proba_b       = sp.empty((K,len(indZ_b)))
    proba_w       = sp.empty((K,len(indZ_w)))
    proba_Cb      = sp.empty((C,len(indZ_b)))
    proba_Cw      = sp.empty((C,len(indZ_w)))
    Z_2d          = sp.zeros((heightImg+2,widthImg+2))-1
    C_2d          = sp.zeros((heightImg+2,widthImg+2))-1
    fzC           = sp.ones((K,C))
    fY            = sp.empty((K,P))
    Lambda        = sp.empty((K,R,R))
    Lambda_inv    = sp.empty((K,R,R))
    mu_param      = sp.empty((K,R))
    activeCluster = list(range(K))
    
    if printEnable:
      plt.ion()
      f, ax = plt.subplots(1,3,figsize=(20,10))
      ax[0].set_title('Cluster labels')
      ax[1].set_title('Class labels')
      ax[2].set_title('Q')                

    ## Sampling
    if nproc == 'all':
      pool=mp.Pool(processes=mp.cpu_count())
    elif nproc>0:
      pool=mp.Pool(nproc)
    start_time = time.time()
    for iterMc in range(Nmc):
      print( "Log: iteration "+str(iterMc+1)+" ("+str(time.time() - start_time)+")" )

      if iterMc >= Nburn :
        self.beta1 = 0.

      # sampling abondances
      for k in activeCluster:
        Lambda_inv[k,:,:] = sp.dot(M.T,M)/s2_est + sp.diag(1/sigma2_est[:,k])
        Lambda[k,:,:] = splin.pinv(Lambda_inv[k,:,:])
        mu_param[k,:] = (1. /sigma2_est[:,k]) * psi_est[:,k]

      for k in activeCluster:
        indK = Z_est == k
        A_est[:,indK] = sp.random.multivariate_normal(sp.zeros(A_est.shape[0]), Lambda[k,:,:],size=indK.sum()).T
        A_est[:,indK] += sp.dot(Lambda[k,:,:], sp.dot(M.T,Y[:,indK])/s2_est + mu_param[k,:,sp.newaxis])

      # sampling cluster label
      Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))

      for k in activeCluster:
        fY[k,:]  = sp.exp( -0.5*sp.sum(sp.log(sigma2_est[:,k])) - 0.5 * sp.sum( (1. /sigma2_est[:,k,sp.newaxis]) * (A_est - psi_est[:,k,sp.newaxis])**2, axis=0) )

      # update black
      Z_b = Z_2d.ravel()[Vb]
      for k in activeCluster:
        proba_b[k,:] = fY[k,indA_b] * sp.exp(self.beta1*sp.sum(Z_b==k,axis=0)) * Q_est[k,C_est[indA_b]]
      proba_b[:,sp.where(sp.sum(proba_b[activeCluster,:],axis=0)<=0)[0]] = 1
      proba_b[activeCluster,:] = proba_b[activeCluster,:]/sp.sum(proba_b[activeCluster,:],axis=0)
      if nproc == 0:
        Z_est[indA_b] = sp.asarray( [sp.random.choice(activeCluster,1,p=proba_b[activeCluster,x]) for x in range(proba_b.shape[1])] ).ravel()
      else:
        Z_est[indA_b] = sp.asarray( pool.map(choice_wrapper, [(activeCluster,1,True,proba_b[activeCluster,x],) for x in range(proba_b.shape[1])]) ).ravel()
      Z_est[sp.logical_not(mask.ravel())] = -1

      # update white
      Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))
      Z_w = Z_2d.ravel()[Vw]
      for k in activeCluster:
        proba_w[k,:] = fY[k,indA_w] * sp.exp(self.beta1*sp.sum(Z_w==k,axis=0)) * Q_est[k,C_est[indA_w]]
      proba_w[:,sp.where(sp.sum(proba_w[activeCluster,:],axis=0)<=0)[0]] = 1
      proba_w[activeCluster,:] = proba_w[activeCluster,:]/sp.sum(proba_w[activeCluster,:],axis=0)
      if nproc == 0:
        Z_est[indA_w] = sp.asarray( [sp.random.choice(activeCluster,1,p=proba_w[activeCluster,x]) for x in range(proba_w.shape[1])] ).ravel()
      else:
        Z_est[indA_w] = sp.asarray( pool.map(choice_wrapper, [(activeCluster,1,True,proba_w[activeCluster,x],) for x in range(proba_w.shape[1])]) ).ravel()
      Z_est[sp.logical_not(mask.ravel())] = -1


      # sampling Q where qij = p(z=i|c=j)
      for k in activeCluster:
        for c in range(C):
          fzC[k,c] = sum(1. for x in range(P) if (Z_est[x]==k and C_est[x]==c))

      for k in activeCluster:
        Q_est[k,:] = sp.random.dirichlet(fzC[k,:] + 1.,1)[:]

      # sampling class label
      # update black
      Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))
      Z_b = Z_2d.ravel()[Vb]
      for k in activeCluster:
        proba_b[k,:] = sp.exp(self.beta1*sp.sum(Z_b==k,axis=0))

      C_2d[1:-1,1:-1] = C_est.reshape((heightImg,widthImg))
      C_b = C_2d.ravel()[Vb]
      for c in range(C):
        proba_Cb[c,:] = sp.exp(self.beta2*sp.sum(C_b==c,axis=0)) * Q_est[Z_est[indA_b],c] * confLabels[indA_b,c] / (sp.sum(proba_b[activeCluster,:] * Q_est[activeCluster,c,sp.newaxis],axis=0))
      proba_Cb = proba_Cb/sp.sum(proba_Cb,axis=0)
      if nproc == 0:
        C_est[indA_b] = sp.asarray( [sp.random.choice(C,1,p=proba_Cb[:,x]) for x in range(proba_Cb.shape[1])] ).ravel()
      else:
        C_est[indA_b] = sp.asarray( pool.map(choice_wrapper, [(C,1,True,proba_Cb[:,x],) for x in range(proba_Cb.shape[1])]) ).ravel()
      C_est[sp.logical_not(mask.ravel())] = -1

      # update white
      Z_w = Z_2d.ravel()[Vw]
      for k in activeCluster:
        proba_w[k,:] = sp.exp(self.beta1*sp.sum(Z_w==k,axis=0))

      C_2d[1:-1,1:-1] = C_est.reshape((heightImg,widthImg))
      C_w = C_2d.ravel()[Vw]
      for c in range(C):
        proba_Cw[c,:] = sp.exp(self.beta2*sp.sum(C_w==c,axis=0)) * Q_est[Z_est[indA_w],c] * confLabels[indA_w,c] / (sp.sum(proba_w[activeCluster,:] * Q_est[activeCluster,c,sp.newaxis],axis=0))
      proba_Cw = proba_Cw/sp.sum(proba_Cw,axis=0)
      if nproc == 0:
        C_est[indA_w] = sp.asarray( [sp.random.choice(C,1,p=proba_Cw[:,x]) for x in range(proba_Cw.shape[1])] ).ravel()
      else:
        C_est[indA_w] = sp.asarray( pool.map(choice_wrapper, [(C,1,True,proba_Cw[:,x],) for x in range(proba_Cw.shape[1])]) ).ravel()
      C_est[sp.logical_not(mask.ravel())] = -1

      # sampling noise variance
      s2_est = 1. / sp.random.gamma(1+P*L/2, 1/(delta_est + 0.5* sp.sum((Y-sp.dot(M,A_est))**2)) )
      
      # sampling cluster mean
      indR = sp.empty((R),dtype=int)
      for k in activeCluster:
        ind = Z_est==k
        nk = float(sp.sum(ind))

        if nk!=0:
          indR[-1]  = sp.random.choice(sp.where(psi_est[:,k]!=0)[0])
          permu = sp.random.permutation(R)
          indR[:-1] = permu[permu!=indR[-1]]
          for r in indR[:-1]:
            psi_est[r,k] = rt.rtnorm(0, 1-(sp.sum(psi_est[indR[:-1],k])-psi_est[r,k]), mu=(1/nk)*sp.sum(A_est[r,ind]), sigma=sp.sqrt(sigma2_est[r,k]/nk), size=1)
          psi_est[indR[-1],k] = 1-sp.sum(psi_est[indR[:-1],k])
        else:
          activeCluster.remove(k)
          fzC[k,:] = 0.

      # sampling cluster variance
      for k in activeCluster:
        ind = Z_est==k
        nk = float(sp.sum(ind))

        param2 = self.gamma + 0.5* sp.sum( (A_est[:,ind] - psi_est[:,k,sp.newaxis])**2, axis=1 )

        sigma2_est[:,k] = 1. / sp.random.gamma(nk/2 +1, 1./param2)
      del param2

      # sampling hyperparameter of noise variance (delta)
      delta_est = 1. / sp.random.gamma(1.,1/s2_est)

      # store sampling
      if self.onlineEstim:
        if iterMc>=Nburn:
          N = float(iterMc - Nburn)
          s2_tab     = (N * s2_tab + s2_est) / (1+N)
          delta_tab  = (N * delta_tab + delta_est) / (1+N)
          A_tab      = (N * A_tab + A_est) / (1+N)
          Q_tab      = (N * Q_tab + Q_est) / (1+N)
          psi_tab    = (N * psi_tab + psi_est) / (1+N)
          sigma2_tab = (N * sigma2_tab + sigma2_est) / (1+N)
          C_tab[C_est,sp.arange(C_est.shape[0])] += 1
          Z_tab[Z_est,sp.arange(Z_est.shape[0])] += 1
      else:
        A_tab[iterMc,:,:]      = A_est
        Z_tab[iterMc,:]        = Z_est
        Q_tab[iterMc,:,:]      = Q_est
        C_tab[iterMc,:]        = C_est
        s2_tab[iterMc]         = s2_est
        psi_tab[iterMc,:,:]    = psi_est
        sigma2_tab[iterMc,:,:] = sigma2_est
        delta_tab[iterMc]      = delta_est

      if printEnable:
        Z_2d[1:-1,1:-1] = Z_est.reshape((heightImg,widthImg))
        ax[0].imshow(Z_2d,interpolation='none')
        C_2d[1:-1,1:-1] = C_est.reshape((heightImg,widthImg))
        ax[1].imshow(C_2d,interpolation='none')
        ax[2].imshow(Q_est[activeCluster,:],interpolation='none')
        f.canvas.draw()
        plt.pause(0.1)


    if nproc != 0:
      pool.close()
      pool.terminate()

    if printEnable:
      plt.close()
      plt.ioff()

    if self.onlineEstim:
      Z_tab = sp.argmax(Z_tab,axis=0)
      Z_tab[sp.logical_not(mask).ravel()] = -1
      C_tab = sp.argmax(C_tab,axis=0)
      C_tab[sp.logical_not(mask).ravel()] = -1

    return A_tab,Z_tab,C_tab+1,Q_tab,psi_tab,sigma2_tab,activeCluster
