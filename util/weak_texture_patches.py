import numpy
import scipy
import math
from skimage.util import view_as_windows as viewW

def im2col_sliding_strided_v2(A, BSZ, stepsize=1):
    return viewW(A, (BSZ[0],BSZ[1])).reshape(-1,BSZ[0]*BSZ[1]).T[:,::stepsize]

def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = numpy.arange(BSZ[0])[:,None]*N + numpy.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = numpy.arange(row_extent)[:,None]*N + numpy.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return numpy.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])

"""
Output params:
nlevel: estimated noise level
th: threshold to extract weak texture patches at the last iteration
num: number of extracted weak texture patches

Input Params:
img: intput single image
patchsize (optional): patch size(default: 7)
decim (optional): decimation factor. Large number = accelerated calculation 
                  (default: 0)
conf (optional): confidence interval to determin the treshold for the 
                 weak texture. usually very close to one (default: 0.99)
itr (optional): number of iterations (default: 3)
"""

def noise_level(img, patchsize = 7, decim = 0, conf = None, itr = 3):
    img = img.astype(numpy.float32)
    if not conf:
        conf = 1-1e-6
    kh = numpy.array([[[-1/2],[0],[1/2]]])
    imgh = scipy.ndimage.correlate(img, kh, mode="nearest")
    imgh = imgh[:,1:len(imgh[1])-1,:]
    imgh = numpy.multiply(imgh, imgh)
    
    kv = numpy.array([[[-1/2]],[[0]],[[1/2]]])
    imgv = scipy.ndimage.correlate(img, kv, mode="nearest")
    imgv = imgv[1:len(imgv)-1,:,:]
    imgv = numpy.multiply(imgv, imgv)
    
    
    Dh = my_convmtx(kh, patchsize, patchsize)
    Dv = my_convmtx(kv, patchsize, patchsize)
    DD = Dh.conj().transpose().dot(Dh)\
         +Dv.conj().transpose().dot(Dv)
    r = numpy.linalg.matrix_rank(DD)
    Dtr = DD.trace(offset=0);
    tau0 = scipy.stats.gamma.ppf(conf,float(r)/2, scale = 2.0 * Dtr / float(r));
    
    nlevel = []
    th = []
    num = []
    for cha in range(img.shape[2]):
        X = im2col_sliding_broadcasting(img[:,:,cha],(patchsize, patchsize))
        Xh = im2col_sliding_broadcasting(imgh[:,:,cha],(patchsize, patchsize-2))
        Xv = im2col_sliding_broadcasting(imgv[:,:,cha],(patchsize-2, patchsize))

        Xtr = numpy.vstack((Xh,Xv)).sum(axis=0)
        if decim > 0:
            XtrX = numpy.vstack((Xtr,X));
            XtrX = sortrows(XtrX.conj().transpose())
            p = math.floor(XtrX.shape[1]/(decim+1))
            p = numpy.array(range(0,p)) * (decim+1)
            Xtr = XtrX[1].take(p, axis=1)
            X = XtrX[2:size(XtrX,1)].take(p, axis=1)
        # Noise level estimation
        tau = math.inf
        if X.shape[1] < X.shape[0]:
            sig2 = 0
        else:
            cov = X.dot((X.conj().T))/(X.shape[1]-1)
            d = numpy.linalg.eigvals(cov)
            d.sort()
            # d = d[::-1]
            sig2 = d[0]
        for i in range(2,itr+1):
            # weak texture selection
            tau = sig2 * tau0
            #p = [1 if pp < tau else 0 for pp in Xtr]
            p = Xtr<tau
            Xtr = Xtr[p]
            X = X[:,p==1]
            
            # noise level estimation
            if X.shape[1] < X.shape[0]:
                break

            cov = X.dot((X.conj().T))/(X.shape[1]-1)
            d = numpy.linalg.eigvals(cov)
            d.sort()
            sig2 = d[0]
        nlevel.append(math.sqrt(sig2))
        th.append(tau)
        num.append(X.shape[1])
    
    return nlevel, th, num
            
def my_convmtx(H, m, n):
    s = H.shape
    T = numpy.zeros((((m-s[0]+1)*(n-s[1]+1)), (m*n)), float)
    k = 0
    for i in range(1,(m-s[0]+2)):
        for j in range(1,(n-s[1]+2)):
            for p in range(1,s[0]+1):
                h_index = p-1
                for x in range((i-1+p-1)*n+(j-1),(i-1+p-1)*n+(j-1)+s[1]):
                    T[k,x] = H.flatten()[h_index]
                    h_index += 1
            k += 1
    return T
    
        
"""
Output parameters:
msk: weak texture mask. 0 and 1 represent non-weak-texture and weak-texture
     respectively

Input Parameters:
img: input single image
th: threshold which is output of NoiseLevel
patchsize (optional): patchsize (default: 7)
"""
def weak_texture_mask(img, th, patchsize=7):
    img = img.astype(numpy.float32) #*(1/255))

    kh = numpy.array([[[-1/2],[0],[1/2]]])
    imgh = scipy.ndimage.correlate(img.astype(numpy.float32), kh, mode="nearest")
    imgh = imgh[:,1:len(imgh[1])-1,:]
    imgh = numpy.multiply(imgh, imgh)
    
    kv = numpy.array([[[-1/2]],[[0]],[[1/2]]])
    imgv = scipy.ndimage.correlate(img.astype(numpy.float32), kv, mode="nearest")
    imgv = imgv[1:len(imgv)-1,:,:]
    imgv = numpy.multiply(imgv, imgv)
    
    
    s = img.shape
    msk = numpy.zeros(s, float)
    
    for cha in range(0,s[2]):
        Xh = im2col_sliding_broadcasting(imgh[:,:,cha],(patchsize, patchsize-2))
        Xv = im2col_sliding_broadcasting(imgv[:,:,cha],(patchsize-2, patchsize))

        Xtr = numpy.vstack((Xh,Xv)).sum(axis=0)
        
        p = [pp < th[cha] for pp in Xtr]
        ind = 0
        for col in range(s[0]-patchsize+1):
            for row in range(s[1]-patchsize+1):
                if p[ind]:
                    msk[col:col+patchsize,row:row+patchsize, cha] = 1
                ind += 1
    return msk