# Generated with SMOP  0.41-beta
from libsmop import *
# MyConvMatrix.m

    
@function
def my_convmtx2(H=None,m=None,n=None,*args,**kwargs):
    varargin = my_convmtx2.varargin
    nargin = my_convmtx2.nargin

    s=size(H)
# MyConvMatrix.m:2
    T=zeros(dot((m - s(1) + 1),(n - s(2) + 1)),dot(m,n))
# MyConvMatrix.m:3
    k=1
# MyConvMatrix.m:4
    for i in arange(1,(m - s(1) + 1)).reshape(-1):
        for j in arange(1,(n - s(2) + 1)).reshape(-1):
            for p in arange(1,s(1)).reshape(-1):
                T[k,arange(dot((i - 1 + p - 1),n) + (j - 1) + 1,dot((i - 1 + p - 1),n) + (j - 1) + 1 + s(2) - 1)]=H(p,arange())
# MyConvMatrix.m:9
            k=k + 1
# MyConvMatrix.m:12
    