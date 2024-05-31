import pandas as pd

import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd

pathData="./1ddata/funcquadraticDiag/"
TVals=[]
TFileNames=[]




for TFile in glob.glob(pathData+"/T*"):
    TFileNames.append(TFile)

    matchT=re.search(r"T(\d+(\.\d+)?)",TFile)

    TVals.append(float(matchT.group(1)))


#sort T files

sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]


def readG(oneTFile):
    """

    :param oneTFile:
    :return: fA, plot of fA
    """
    matchT=re.search(r"T(\d+(\.\d+)?)",oneTFile)
    TVal=float(matchT.group(1))
    GAAFile=oneTFile+"/G.csv"
    GAAData=pd.read_csv(GAAFile,header=None)
    GAAMat=GAAData.to_numpy()
    N,_=GAAMat.shape
    f=[]
    for i in range(0,N):
        supDiag=np.diagonal(GAAMat,offset=i)
        f.append(np.sum(supDiag)/supDiag.size)

    outfFile=oneTFile+"/f.csv"
    f=np.array(f)

    np.savetxt(outfFile, f, delimiter=',')



for oneTFile in sortedTFiles:
    readG(oneTFile)


#compare G
fArray=[]
for oneTFile in sortedTFiles:
    pathfTmp=oneTFile+"/f.csv"
    dfTmp=pd.read_csv(pathfTmp,header=None)
    arrTmp=dfTmp.to_numpy()

    fArray.append(arrTmp.T[0,:])


#each row is G for one temperature, each column is the G with the same distance under different tempeatures
fArray=np.array(fArray)

N=fArray[0,:].size

def plotG(j):
    """

    :param j: distance
    :return: correlation function of xA
    """
    colTmp=fArray[:,j]
    plt.plot(sortedTVals,colTmp,color="blue")
    plt.scatter(sortedTVals,colTmp,color="red")
    plt.xlabel("T")
    plt.ylabel("$G($"+str(j)+"$)$")
    plt.ylim((-0.5,0.5))


    plt.savefig(pathData+"/GA"+str(j)+".pdf")

    plt.close()



def plotr(j):
    """

    :param j: distance
    :return: correlation coefficient of xA
    """
    colTmp=fArray[:,j]
    col0=fArray[:,0]
    rhoTmp=colTmp/col0
    plt.plot(sortedTVals,rhoTmp,color="green")
    plt.scatter(sortedTVals,rhoTmp,color="magenta")
    plt.xlabel("T")
    plt.ylabel("$\\rho($"+str(j)+"$)$")
    plt.ylim((-1,1))


    plt.savefig(pathData+"/rhoAA"+str(j)+".pdf")

    plt.close()



plotG(int(N/2))
plotr(int(N/2))
