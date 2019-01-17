# Generated with SMOP  0.41-beta
from libsmop import *
# NoiseLevel.m

    
@function
def NoiseLevel(img=None,patchsize=None,decim=None,conf=None,itr=None,*args,**kwargs):
    varargin = NoiseLevel.varargin
    nargin = NoiseLevel.nargin

    if (logical_not(exist('itr','var'))):
        itr=3
# NoiseLevel.m:3
    
    if (logical_not(exist('conf','var'))):
        conf=1 - 1e-06
# NoiseLevel.m:6
    
    if (logical_not(exist('decim','var'))):
        decim=0
# NoiseLevel.m:9
    
    if (logical_not(exist('patchsize','var'))):
        patchsize=7
# NoiseLevel.m:12
    
    kh=concat([- 1 / 2,0,1 / 2])
# NoiseLevel.m:14
    imgh=imfilter(img,kh,'replicate')
# NoiseLevel.m:15
    imgh=imgh(arange(),arange(2,size(imgh,2) - 1),arange())
# NoiseLevel.m:16
    imgh=multiply(imgh,imgh)
# NoiseLevel.m:17
    kv=kh.T
# NoiseLevel.m:18
    imgv=imfilter(img,kv,'replicate')
# NoiseLevel.m:19
    imgv=imgv(arange(2,size(imgv,1) - 1),arange(),arange())
# NoiseLevel.m:20
    imgv=multiply(imgv,imgv)
# NoiseLevel.m:21
    Dh=my_convmtx2(kh,patchsize,patchsize)
# NoiseLevel.m:22
    Dv=my_convmtx2(kv,patchsize,patchsize)
# NoiseLevel.m:23
    DD=dot(Dh.T,Dh) + dot(Dv.T,Dv)
# NoiseLevel.m:24
    r=rank(DD)
# NoiseLevel.m:25
    Dtr=trace(DD)
# NoiseLevel.m:26
    tau0=gaminv(conf,double(r) / 2,dot(2.0,Dtr) / double(r))
# NoiseLevel.m:27
    #{
    eg=eig(DD)
# NoiseLevel.m:29
    tau0=gaminv(conf,double(r) / 2,dot(2.0,eg(dot(patchsize,patchsize))))
# NoiseLevel.m:30
    #}
    for cha in arange(1,size(img,3)).reshape(-1):
        X=im2col(img(arange(),arange(),cha),concat([patchsize,patchsize]))
# NoiseLevel.m:33
        Xh=im2col(imgh(arange(),arange(),cha),concat([patchsize,patchsize - 2]))
# NoiseLevel.m:34
        Xv=im2col(imgv(arange(),arange(),cha),concat([patchsize - 2,patchsize]))
# NoiseLevel.m:35
        Xtr=sum(vertcat(Xh,Xv))
# NoiseLevel.m:37
        if (decim > 0):
            XtrX=vertcat(Xtr,X)
# NoiseLevel.m:39
            XtrX=sortrows(XtrX.T).T
# NoiseLevel.m:40
            p=floor(size(XtrX,2) / (decim + 1))
# NoiseLevel.m:41
            p=dot(concat([arange(1,p)]),(decim + 1))
# NoiseLevel.m:42
            Xtr=XtrX(1,p)
# NoiseLevel.m:43
            X=XtrX(arange(2,size(XtrX,1)),p)
# NoiseLevel.m:44
        ##### noise level estimation #####
        tau=copy(Inf)
# NoiseLevel.m:47
        if (size(X,2) < size(X,1)):
            sig2=0
# NoiseLevel.m:49
        else:
            cov=dot(X,X.T) / (size(X,2) - 1)
# NoiseLevel.m:51
            d=eig(cov)
# NoiseLevel.m:52
            sig2=d(1)
# NoiseLevel.m:53
        for i in arange(2,itr).reshape(-1):
            ##### weak texture selectioin #####
            tau=dot(sig2,tau0)
# NoiseLevel.m:58
            p=(Xtr < tau)
# NoiseLevel.m:59
            Xtr=Xtr(arange(),p)
# NoiseLevel.m:60
            X=X(arange(),p)
# NoiseLevel.m:61
            if (size(X,2) < size(X,1)):
                break
            cov=dot(X,X.T) / (size(X,2) - 1)
# NoiseLevel.m:67
            d=eig(cov)
# NoiseLevel.m:68
            sig2=d(1)
# NoiseLevel.m:69
        nlevel[cha]=sqrt(sig2)
# NoiseLevel.m:71
        th[cha]=tau
# NoiseLevel.m:72
        num[cha]=size(X,2)
# NoiseLevel.m:73
    