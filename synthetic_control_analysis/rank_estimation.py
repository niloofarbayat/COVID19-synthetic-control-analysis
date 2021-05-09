import numpy as np
import pandas as pd

def Chebyshev_vTAv(A, v, deg):
    
    n = len(v)
    y = np.zeros((deg + 1, 1));
    vkm1 = np.zeros((n, 1));

    vk = v;
    y[0, 0] = 1;

    for k in range(deg):
        
        #accumulation of vector.
        scal = 2;
        if k == 0:
            scal = 1;

        vkp1 = scal* (A @ vk) - vkm1
        vkm1 = vk
        vk = vkp1
        y[k+1, 0] = (v.T @ vk)
        
    return y

def find_gap(xx, yy, n):
    
    npts = len(yy)
    # tolerance for high values
    tol1 = (npts+1)*0.5
    # smooth the curve
    p = 3;   # p = 0 ----> no smoothing
    zz = np.abs(yy)
    for ii in range(2, p+1):
        zz[:npts-ii] = zz[:npts-ii] + yy[ii:npts]

    yy = zz/(p+1)
    # derivatives are often better at spotting gap:
    tt = np.zeros((npts, 1))
    tt[:npts-1] = yy[1:npts]-yy[0:npts-1]  # (xx(2)-xx(1));
    # get 1st point of increase
    ysum = 0;
    for i in range(npts):
        # spot first time derivative turns>0
        ysum = ysum + yy[i]
        # discard high values
        if yy[i]>tol1:
            continue

        if tt[i] >= -0.01:
            indx1 = i
            break

    ysum = ysum/(npts+1)
    
    ## spot first time yy becomes significant again
    tol  = (1 - ysum)*0.25
    indx2 = [];

    print("ysum")

    print(ysum, indx1)
    for i in range(indx1+1, npts):
        ## spot first time derivative becomes>0
        if yy[i] >= tol:
            indx2 = i
            break

    if not indx2:
        indx2=indx1+1

    gap = (xx[indx1]+xx[indx2])/2

    print("indx1 indx2")
    print(indx1, indx2, yy[indx1], yy[indx2], xx[indx1], xx[indx2])
    
    return gap

def kpmchebdos_new(Y, Npts, lm, lM, damping):
    ## To do
    h = 2/(Npts+1)
    eps = 2**(-52)
    #eps = 2 ** (-20)
    
    pts = np.concatenate((np.arange(-1+h, eps, h * 0.5), np.arange(h,1-h + eps, 2*h)))

    deg, nvec = Y.shape
    
    ctr = (lm+lM)/2;
    wid = (lM-lm)/2 + 2*eps

    mu = np.zeros((deg,1))
    
    for m in range(nvec):
        wk   = Y[:,m]
        thetJ = np.pi/(deg+2)
        thetL = np.pi/(deg+1)
        a1 = 1/(deg+2);
        a2 = np.sin(thetJ)
        
        for k in range(deg):
            if damping == 0:
                jac = 1
            elif damping  == 1:
                # Note: slightly simpler formulas for jackson:
                jac = a1*np.sin((k+1)*thetJ)/a2+(1-(k+1)*a1)*np.cos(k*thetJ);
                # Lanczos sigma-damping:
            elif damping == 2:
                jac = 1
                if k > 0:
                    jac = np.sin(k*thetL)/(k*thetL)
            mu[k] = mu[k]+jac*(wk[k])


    # Plot
    mu = 2*mu/ (nvec*np.pi)  # scaling in formula
    mu[0] = mu[0]/2         # first term is different

    xx  = ctr + wid*pts
    y2 = TcPlt(mu, pts)
    yy  = y2
    # yy = (nptsC/nptsR) *yy /sum(yy);
    yy  = yy / (np.sum(yy)*(xx[2]-xx[1]))
    
    return xx,yy

def TcPlt(mu, xi):
    n = xi.shape[0]
    m = len(mu)
    # compute p(xi)
    yi = np.zeros((n,1))
    vkm1 = np.zeros((n,1))
    vk   = np.ones((n,1))
    #print((xi.reshape(-1, 1) * vk).shape)
    for k in range(m):
    # accumulation of vector.
        yi = yi + mu[k, 0] * vk
        scal = 2
        if k == 0:
            scal = 1;
        vkp1 = scal *xi.reshape(-1, 1) * vk- vkm1
        vkm1 = vk
        vk = vkp1
    
    
    yi = yi/ np.sqrt(1 - xi.reshape(-1, 1)**2)

    return yi

def jac2Plt(m, ab, damping, xi = 0):
    alpha1 = ab[0]
    alpha2 = ab[1]
    thetJ = np.pi/(m+2)
    thetL = np.pi/(m+1)
    
    a1 = 1/(m+2)
    a2 = np.sin(thetJ)
    beta1 = np.arccos(alpha1)
    beta2 = np.arccos(alpha2)
    mu = np.zeros(m+1)
    
    for k in range(0, m+1):
        # generate Jackson coefficients
        if damping == 0:
            
            jac = 1;
            
        elif damping  == 1:
            # Note: slightly simpler formulas for jackson: 
            jac = a1*np.sin((k+1)*thetJ)/a2 + (1-(k+1)*a1)*np.cos(k*thetJ)
            
            #-------------------- Lanczos sigma-damping: 
        elif damping == 2:
            jac = 1
            if k > 0:
                jac = np.sin(k*thetL)/(k*thetL)

        if k == 0:
            mu[k] = -jac*(beta2-beta1)/np.pi;   
        else:
            mu[k] = -2*jac*(np.sin(k*beta2)-np.sin(k*beta1))/(np.pi*k); 
    
    # To do: for plotting
    return mu,[] 


def KPM_DOS_and_Rank(A,lmin,lmax,deg,nvecs,Npts,AutoGap):
    
    #Initialization
    
    n = A.shape[1]
    dd = (lmax - lmin)/2
    cc = (lmax + lmin)/2
    B = (A - cc*np.diag([1 for i in range(n)]))/dd
    z1 = np.zeros((nvecs, 1))
    zz = np.zeros((nvecs, 1))
    
    # Form v'Tk(A)v for nvecs starting vectors
    Y = np.zeros((deg+1, nvecs))
    
    for i in range(nvecs):
        v = np.random.normal(size = (n, 1)) 
        # v = np.random.rand(n, 1)
        #v = np.ones((n, 1))
        v = v/np.linalg.norm(v, 2)
        z = Chebyshev_vTAv(B, v, deg)
        Y[:,i] = z.T

    #################Debug
    # pd.DataFrame(Y).to_csv("../../Desktop/research/rank_estimation (1)/Y.csv")
    # print(Y[50])
    #Y = pd.read_csv("../../Desktop/research/rank_estimation (1)/Y.csv", index_col = 0).values
    #################
    # Find DOS and Threshold
    
    if AutoGap:
        ## To do
        damping  = 1; # type of damping: 0 - no damping; 1 - Jackson and 2 - Lanczos
        [xx, yy] = kpmchebdos_new(Y,Npts,lmin,lmax,damping);
        # locate gap
        eps = find_gap(xx,yy,n)
    else:
        xx = np.zeros((Npts,1))
        yy = np.zeros((Npts,1))
        eps = 1.35e5;  #set a threshold
    
    ab0 = np.array([eps, lmax])
    ab = (ab0 - np.array([cc, cc]))/dd
    cnt_est = 0
    mu,_ = jac2Plt(deg, ab, 2)
    

    for l in range(nvecs):
        vk = Y[:,l]
        t = sum(mu * vk)
        #print(t)
        z1[l] = t*n
        cnt_est = (cnt_est+t)
        zz[l] = n*(cnt_est/(l+1))
    
    r = zz[nvecs-1];
    
    return r,eps,zz,z1,xx,yy

def estimate_rank(X):
    deg   = 100;
    eps = 0.001;
    nvecs = 30;
    AutoGap = 1;
    Npts = 100;

    n = X.shape[1]
    d = np.linalg.svd(X)[1]
    A = X.T @ X
    d = d.T * d
    d = np.sort(d)
    lmin = d[0] - 0.00001
    lmax = d[-1] + 0.00001

    r,g,zz,z1,xx,yy = KPM_DOS_and_Rank(A,lmin,lmax,deg,nvecs,Npts,AutoGap)
    print(r, g)
    #return int(round(r[0]))
    # return r,g,zz,z1,xx,yy
    return np.sum(d > (g))





    
    


    