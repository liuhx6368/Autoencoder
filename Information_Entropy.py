import sys,os,math
import numpy as np

def Normalize(b):
    for i in range(len(b)):
        Max=max(b[i])
        Min=min(b[i])
        d=Max-Min
        for j in range(len(b[i])):
            b[i][j]=(b[i][j]-Min)/d
    return b

def prepare(X,m):
    X=X.T
    X_float=Normalize(X)
    X_int=np.zeros((len(X),len(X[0])),dtype=float)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_int[i][j]=int(X[i][j]*m)
    return X_int

def H_X(v,y):
    H=0
    for x in set(v):
        pi=len([v[i] for i in range(len(v)) if v[i]==x])/len(v)
        try:
            y_x=[y[i] for i in range(len(v)) if v[i]==x]
        except IndexError:
            continue
        h=0
        for i in set(y_x):
            pij=len([j for j in range(len(y_x)) if y_x[j]==i])/len(y)
            h=h+pij*math.log(pij/pi,2)
        H=H+h
    return H

def entropy(n,m):
    X_ini_1=np.load('Cluster_'+n+'_hop.npy')
    X_ini_2=np.load('Cluster_'+n+'_ini.npy')
    y=np.hstack(([0 for i in range(len(X_ini_1))],[1 for i in range(len(X_ini_2))]))
    X=np.vstack((X_ini_1,X_ini_2))
    X_p=prepare(X,m)
    H0=H_X(np.zeros(len(y)),y)
    result=[H_X(v,y)/H0 for v in X_p]
    return result

result=entropy(sys.argv[1],9.9999999)
print(result)

