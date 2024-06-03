# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:53:12 2024

@author: User
"""

import os
import pandas as pd
pd.__version__
import sys
version = 1.0
sourceIP = '135.0.132.145'

#See README.txt for documentation

# V1.0 - FGC-17

def IsolateData(df):
    featureNames = df.columns.tolist()
    requiredNames = ['Time', 'Length']
    if all(item in featureNames for item in requiredNames):
        # print("Found all required features ...")
        # print("\nFound Featutures:", featureNames)
        pktLength = df['Length']

        isolatedData = pd.DataFrame({'Length': pktLength})
        return isolatedData
    else:
        print("\nThe specified .csv is missing one of the following paramerters, be sure they are all spelt correctly (cap sensitive);")
        print("\nTime\nSource\nDestination\nLength\n")
        sys.exit()