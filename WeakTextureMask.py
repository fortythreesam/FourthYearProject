# Generated with SMOP  0.41-beta
from libsmop import *
# WeakTextureMask.m

    
@function
def WeakTextureMask(img=None,th=None,patchsize=None,*args,**kwargs):
    varargin = WeakTextureMask.varargin
    nargin = WeakTextureMask.nargin

    if (logical_not(exist('patchsize','var'))):
        patchsize=7
# WeakTextureMask.m:3
    
    kh=concat([- 1 / 2,0,1 / 2])
# WeakTextureMask.m:5
    imgh=imfilter(img,kh,'replicate')
# WeakTextureMask.m:6
    imgh=imgh(arange(),arange(2,size(imgh,2) - 1),arange())
# WeakTextureMask.m:7
    imgh=multiply(imgh,imgh)
# WeakTextureMask.m:8
    kv=kh.T
# WeakTextureMask.m:9
    imgv=imfilter(img,kv,'replicate')
# WeakTextureMask.m:10
    imgv=imgv(arange(2,size(imgv,1) - 1),arange(),arange())
# WeakTextureMask.m:11
    imgv=multiply(imgv,imgv)
# WeakTextureMask.m:12
    s=size(img)
# WeakTextureMask.m:13
    msk=zeros(s)
# WeakTextureMask.m:14
    for cha in arange(1,s(3)).reshape(-1):
        m=im2col(img(arange(),arange(),cha),concat([patchsize,patchsize]))
# WeakTextureMask.m:16
        m=zeros(size(m))
# WeakTextureMask.m:17
        Xh=im2col(imgh(arange(),arange(),cha),concat([patchsize,patchsize - 2]))
# WeakTextureMask.m:18
        Xv=im2col(imgv(arange(),arange(),cha),concat([patchsize - 2,patchsize]))
# WeakTextureMask.m:19
        Xtr=sum(vertcat(Xh,Xv))
# WeakTextureMask.m:21
        p=(Xtr < th(cha))
# WeakTextureMask.m:23
        ind=1
# WeakTextureMask.m:24
        for col in arange(1,s(2) - patchsize + 1).reshape(-1):
            for row in arange(1,s(1) - patchsize + 1).reshape(-1):
                if (p(ind) > 0):
                    msk[arange(row,row + patchsize - 1),arange(col,col + patchsize - 1),cha]=1
# WeakTextureMask.m:28
                ind=ind + 1
# WeakTextureMask.m:30
    
    return msk
    
if __name__ == '__main__':
    pass
    