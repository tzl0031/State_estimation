import pdb
import g01
import matplotlib.pyplot as plt
global Gm,F0,Np,CR,D
Gm=500
F0=0.5
Np=130
CR=0.2
G=1
D=13
import numpy as np
V=np.zeros((Np,D))
U=np.zeros((Np,D))
XG_next=np.zeros((Np,D))
value=np.zeros(Np)
cv=np.zeros(Np)
cv[:]=1000
##############产生初始种群############
from numpy import random
max0=1
max1=100
min0=0
min1=0
X0=random.random(size=(Np,D))
for i in range(9):
    X0[:,i]=X0[:,i]*(max0-min0)+min0
for i in range(9,12):
    X0[:,i]=X0[:,i]*(max1-min1)+min1
X0[:,12]=X0[:,12]*(max0-min0)+min0
XG=X0


while G<=Gm:
    print G
    ####################变异操作###################
    for i in range(1,Np):
        li=range(Np)
        random.shuffle(li)
        dx=li
        j=dx[0]
        k=dx[1]
        p=dx[2]
        #要保证与i不同
        if j==i:
            j=dx[3]
        elif k==i:
            k=dx[3]
        elif p==i:
            p=dx[3]
        #变异操作
        suanzi=numpy.math.exp(1 - numpy.float(Gm) / (Gm + 1 - G))
        F=F0*(2**suanzi)
        mutant=XG[p]+F*(XG[j]-XG[k])
        for j in range(9):
            if mutant[j]>min0 and mutant[j]<max0:
                V[i,j]=mutant[j]
            else:
                V[i,j]=(max0-min0)*random.rand()+min0
        for j in range(9,12):
            if mutant[j]>min1 and mutant[j]<max1:
                V[i,j]=mutant[j]
            else:
                V[i,j]=(max1-min1)*random.rand()+min1
        if mutant[12]>min0 and mutant[12]<max0:
                V[i,12]=mutant[12]
        else:
            V[i,12]=(max0-min0)*random.rand()+min0
    #######################交叉操作#####################
    
    for i in range(Np):
        randx=range(D)
        random.shuffle(randx)
        for j in range(D):
            if random.rand()>CR and randx[0]!=j:
                U[i,j]=XG[i,j]
            else:
                U[i,j]=V[i,j]
    ########################选择操作######################
    for i in range(Np):
        f1=g01.f(XG[i])
        f2=g01.f(U[i])
        g1=g01.g(XG[i])
        g2=g01.g(U[i])
        if g1==0 and g2==0:
            if f1<f2:
                XG_next[i]=XG[i]
            else:
                XG_next[i]=U[i]
        elif g1==0 or g2==0:
            if g1==0:
                XG_next[i]=XG[i]
            else:
                XG_next[i]=U[i]
        elif g1!=0 and g2!=0:
            if g1<g2:
                XG_next[i]=XG[i]
            else:
                XG_next[i]=U[i]
    XG=XG_next
    G=G+1
for i in range(Np):
    cv[i]=g01.g(XG[i])
    if numpy.abs(cv[i] - 0)<0.05:
        value[i]=g01.f(XG[i])
best_value=numpy.min(value)
temp=value.tolist()
pos_min=temp.index(numpy.min(temp))
print best_value
best_vector=XG[pos_min]
print best_vector


---------------------
作者：Liujing_y
来源：CSDN
原文：https://blog.csdn.net/Liujing_y/article/details/78437979
版权声明：本文为博主原创文章，转载请附上博文链接！