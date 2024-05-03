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

pathData="./data/"

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
        print(TFile)
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

    UAllDir=oneTFile+"/UAll/*"
    xAllDir=oneTFile+"/xAll/*"

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
            startxAllVals.append(int(matchxAllStart))

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


