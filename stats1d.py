import xml.etree.ElementTree as ET
import numpy as np
import glob
import sys
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt
# from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime

#this script computes statistics for 1d, plots evolution

moveNumInOneFlush=3000

pathData="./1ddata/"

aVals=[]
aFileNames=[]
TValsForAll_a=[]
TFileNamesForAll_a=[]
sortedTFilesForAll_a=[]
sortedTValsForAll_a=[]

for a_file in glob.glob(pathData+"/a*"):
    #first search a values
    aFileNames.append(a_file)
    match_a=re.search(r"a(\d+(\.\d+)?)",a_file)
    aVals.append(float(match_a.group(1)))
    #for each a, search T values
    TFilesTmp=[]
    TValsTmp=[]
    for TFile in glob.glob(a_file+"/T*"):
        TFilesTmp.append(TFile)
        # print(TFile)
        matchT=re.search(r"T(\d+(\.\d+)?)",TFile)
        TValsTmp.append(float(matchT.group(1)))

    TFileNamesForAll_a.append(TFilesTmp)
    TValsForAll_a.append(TValsTmp)

#sort T files for each a
for j in range(0,len(aVals)):
    T_indsTmp=np.argsort(TValsForAll_a[j])
    TValsTmp=TValsForAll_a[j]
    sortedTValsTmp=[TValsTmp[i] for i in T_indsTmp]
    sortedTValsForAll_a.append(sortedTValsTmp)

    TFilesTmp=TFileNamesForAll_a[j]
    sortedTFilesTmp=[TFilesTmp[i] for i in T_indsTmp]
    sortedTFilesForAll_a.append(sortedTFilesTmp)





def parseSummaryBeforeEq(summaryFile):
    '''

    :param summaryFile: summary.txt
    :return: same, loopNumBeforeEq,lag,lastFileNum
    '''
    fptr=open(summaryFile,"r")
    contents=fptr.readlines()

    same=False
    loopNumBeforeEq=-1
    lastFileNum=-1
    lag=-1
    for line in contents:

        matchSame=re.search(r"same:\s*(\d+)",line)
        if matchSame:
            same=int(matchSame.group(1))

        matchLoopNum=re.search(r"total loop number\s*:\s*(\d+)",line)
        if matchLoopNum:
            loopNumBeforeEq=int(matchLoopNum.group(1))

        matchLag=re.search(r"lag=\s*(\d+)",line)

        if matchLag:
            lag=int(matchLag.group(1))


        matchLastFileNum=re.search(r"lastFileNum=\s*(\d+)",line)
        if matchLastFileNum:
            lastFileNum=int(matchLastFileNum.group(1))

    return same, loopNumBeforeEq,lag,lastFileNum


def searchsummaryAfterEqFile(oneTFile):
    """

    :param oneTFile: one T directory
    :return: a list containing the summaryAfterEq.txt file, the list will be length 0 if the
    file does not exist
    """
    file=glob.glob(oneTFile+"/summaryAfterEq.txt")

    return file


def parseAfterEqFile(oneTFile):
    """

    :param oneTFile: one T directory
    :return: a list containing the summaryAfterEq.txt file, and the total loop number after eq
    """
    fileList=searchsummaryAfterEqFile(oneTFile)
    loopNumAfterEq=-1
    if len(fileList)!=0:
        fileName=fileList[0]
        fptr=open(fileName,"r")
        contents=fptr.readlines()
        for line in contents:
            matchNum=re.search(r"total loop number\s*:\s*(\d+)",line)
            if matchNum:
                loopNumAfterEq=int(matchNum.group(1))


    return fileList,loopNumAfterEq


def UAndxFilesSelected(oneTFile):
    """

    :param oneTFile: one T directory
    :return: U files and x files to be parsed
    """

    smrFile=oneTFile+"/summary.txt"
    same, loopNumBeforeEq,lag,lastFileNum=parseSummaryBeforeEq(smrFile)
    fileAfterEqList,loopNumAfterEq=parseAfterEqFile(oneTFile)
    fileNumSelected=0#files' numbers to be parsed
    if same==1:
        fileNumSelected=1
    else:
        if len(fileAfterEqList)!=0:
            loopNumToInclude=moveNumInOneFlush*lastFileNum+loopNumAfterEq
            fileNumSelected=int(np.ceil(loopNumToInclude/moveNumInOneFlush))
        else:
            fileNumSelected=lastFileNum

    UAllDir=oneTFile+"/UAll/*.xml"
    xAllDir=oneTFile+"/xAll/*.xml"

    inUAllFileNames=[]
    startUAllVals=[]
    for file in glob.glob(UAllDir):
        inUAllFileNames.append(file)
        matchUStart=re.search(r"loopStart(-?\d+(\.\d+)?)loopEnd",file)
        if matchUStart:
            startUAllVals.append(int(matchUStart.group(1)))

    start_U_inds=np.argsort(startUAllVals)

    sortedUAllFileNames=[inUAllFileNames[ind] for ind in start_U_inds]


    inxAllFileNames=[]
    startxAllVals=[]
    for file in glob.glob(xAllDir):
        inxAllFileNames.append(file)
        matchxAllStart=re.search(r"loopStart(-?\d+(\.\d+)?)loopEnd",file)
        if matchxAllStart:
            startxAllVals.append(int(matchxAllStart.group(1)))

    start_xAll_inds=np.argsort(startxAllVals)
    sortedxAllFileNames=[inxAllFileNames[ind] for ind in start_xAll_inds]

    retUAllFileNames=sortedUAllFileNames[-fileNumSelected:]
    retxAllFileNames=sortedxAllFileNames[-fileNumSelected:]


    return same, retUAllFileNames,retxAllFileNames,lag,fileNumSelected




def parseUFile(UFileName):
    """

    :param UFileName: xml file containing U
    :return: values of U in this file
    """

    tree=ET.parse(UFileName)
    root = tree.getroot()
    vec=root.find("vec")
    vec_items=vec.findall('item')
    vecValsAll=[float(item.text) for item in vec_items]
    # vecValsAll=np.array(vecValsAll)

    return vecValsAll


def parsexFile(xFileName):
    """

    :param xFileName: xml file containing x vectors
    :return: all vectors in this xml file
    """
    tree=ET.parse(xFileName)
    root = tree.getroot()
    first_level_items = root.find('vecvec').findall('item')
    vectors=[]
    for item in first_level_items:
        oneVec=[float(value.text) for value in item.findall('item')]
        vectors.append(oneVec)

    return np.array(vectors)


def combineValues(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return: combined values of U and x from each file, names of the parsed files
    """
    same, retUAllFileNames,retxAllFileNames,lag,fileNumSelected=UAndxFilesSelected(oneTFile)


    UVecValsCombined=parseUFile(retUAllFileNames[0])

    for file in retUAllFileNames[1:]:
        UVecValsCombined+=parseUFile(file)

    xVecVecCombined=parsexFile(retxAllFileNames[0])


    for file in retxAllFileNames[1:]:
        xVecVecNext=parsexFile(file)

        xVecVecCombined=np.r_[xVecVecCombined,xVecVecNext]

    return same, UVecValsCombined,xVecVecCombined,lag,fileNumSelected


def meanAndVarForScalar(vec):
    """

    :param vec: a vector of float
    :return: mean, half length of confident interval
    """

    meanVal=np.mean(vec)
    varVal=np.var(vec,ddof=1)
    hfLength=1.96*np.sqrt(varVal/len(vec))

    return meanVal,hfLength


outRoot=pathData


def diagnosticsAndStats(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return: diagnostic plots and observable values
    """
    tOneFileStart=datetime.now()
    TTmpMatch=re.search(r"T(\d+(\.\d+)?)",oneTFile)
    if TTmpMatch:
        TTmp=float(TTmpMatch.group(1))
    same, UVecValsCombined,xVecVecCombined,lag,fileNumSelected=combineValues(oneTFile)

    ##############diagnostics: not identical values ################################################################
    if same==0:
        #diagnostics for U



        USelected=UVecValsCombined[::lag]


        meanU=np.mean(USelected)
        varU=np.var(USelected,ddof=1)
        sigmaU=np.sqrt(varU)
        # print("varU="+str(varU))
        hfIntervalU=1.96*np.sqrt(varU/len(USelected))



        #diagnostics of U
        nbins=500

        #histogram of U
        fig=plt.figure()
        axU=fig.add_subplot()
        (n0,_,_)=axU.hist(USelected,bins=nbins)
        meanU=np.round(meanU,4)
        sigmaU=np.round(sigmaU,4)
        axU.set_title("T="+str(np.round(TTmp,3)))
        axU.set_xlabel("$U$")
        axU.set_ylabel("#")
        xPosUText=(np.max(USelected)-np.min(USelected))*1/2+np.min(USelected)
        yPosUText=np.max(n0)*2/3
        axU.text(xPosUText,yPosUText,"mean="+str(meanU)+"\nsd="+str(sigmaU)+"\nlag="+str(lag))
        plt.axvline(x=meanU,color="red",label="mean")
        axU.text(meanU*1.1,0.5*np.max(n0),str(meanU)+"$\pm$"+str(sigmaU),color="red")
        axU.hlines(y=0,xmin=meanU-sigmaU,xmax=meanU+sigmaU,color="green",linewidth=15)

        plt.legend(loc="best")

        EHistOut="T"+str(TTmp)+"UHist.png"
        plt.savefig(oneTFile+"/"+EHistOut)

        plt.close()
        ### test normal distribution for mean U
        USelectedAll=USelected

        #block mean
        def meanPerBlock(length):
            blockNum=int(np.floor(len(USelectedAll)/length))
            UMeanBlock=[]
            for blkNum in range(0,blockNum):
                blkU=USelectedAll[blkNum*length:(blkNum+1)*length]
                UMeanBlock.append(np.mean(blkU))
            return UMeanBlock

        fig=plt.figure(figsize=(20,20))
        fig.tight_layout(pad=5.0)
        lengthVals=[10,20,50,100]
        for i in range(0,len(lengthVals)):
            l=lengthVals[i]
            UMeanBlk=meanPerBlock(l)
            ax=fig.add_subplot(2,2,i+1)
            (n,_,_)=ax.hist(UMeanBlk,bins=100,color="aqua")
            xPosTextBlk=(np.max(UMeanBlk)-np.min(UMeanBlk))*1/7+np.min(UMeanBlk)
            yPosTextBlk=np.max(n)*3/4
            meanTmp=np.mean(UMeanBlk)
            meanTmp=np.round(meanTmp,3)
            sdTmp=np.sqrt(np.var(UMeanBlk))
            sdTmp=np.round(sdTmp,3)
            ax.set_title("L="+str(l))
            ax.text(xPosTextBlk,yPosTextBlk,"mean="+str(meanTmp)+", sd="+str(sdTmp))
        fig.suptitle("T="+str(TTmp))
        plt.savefig(oneTFile+"/T"+str(TTmp)+"UBlk.png")
        # plt.savefig(EBlkMeanDir+"/T"+str(TTmp)+"EBlk.png")
        plt.close()

        plt.figure()

    #diagnostics of x
    xVecVecSelected=xVecVecCombined[::lag,:]
    xValsForEachPosition=[]
    _,nColx=xVecVecSelected.shape
    for j in range(0,nColx):
        xFor1Point=xVecVecSelected[:,j]
        xValsForEachPosition.append(xFor1Point)

    fig=plt.figure(figsize=(20,160))
    fig.tight_layout(pad=5.0)
    x_vertical_distance = 0.9
    xTicks=list(np.arange(0,nColx))
    for j in range(0,nColx):
        axx=fig.add_subplot(nColx,1,j+1,sharex=axx if j != 0 else None)
        xValsTmp=xValsForEachPosition[j]
        xMeanTmp=np.mean(xValsTmp)
        xVarTmp=np.var(xValsTmp,ddof=1)
        xSigmaTmp=np.sqrt(xVarTmp)
        xHfInterval=1.96*np.sqrt(xVarTmp/len(xValsTmp))
        nbins=500
        (n,_,_)=axx.hist(xValsTmp,bins=nbins)
        xMeanTmp=np.round(xMeanTmp,4)
        xSigmaTmp=np.round(xSigmaTmp,4)
        hPosText=(np.max(xValsTmp)-np.min(xVarTmp))*1/7+np.min(xValsTmp)
        vPosText=np.max(n)*3/4
        axx.set_title("position "+str(j)+", T="+str(np.round(TTmp,3)))
        plt.axvline(x=xMeanTmp,color="red",label="mean")
        axx.text(xMeanTmp*1.1,0.5*np.max(n),str(xMeanTmp)+"$\pm$"+str(xSigmaTmp),color="red")
        # axx.hlines(y=0,xmin=xMeanTmp-xHfInterval,xmax=xMeanTmp+xHfInterval,color="green",linewidth=15)
        plt.legend(loc="best")
        axx.set_xticks(xTicks)
        axx.set_ylabel("#")
        axx.set_xlabel("position")
        xHistOut="T"+str(TTmp)+"xHist.pdf"
    plt.subplots_adjust(hspace=x_vertical_distance)
    plt.savefig(oneTFile+"/"+xHistOut)
    plt.close()












tStatsStart=datetime.now()
for item in TFileNamesForAll_a:
    for oneTFile in item:
        diagnosticsAndStats(oneTFile)

tStatsEnd=datetime.now()
print("stats total time: ",tStatsEnd-tStatsStart)

