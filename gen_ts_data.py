#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:45:50 2024

@author: williamsommers
"""

# William Sommers
# HiveMQ Technical Account Manager (TAM)


import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(444)
np.set_printoptions(precision=2)  # Output decimal fmt.


# Synthesized simulated current over the motor life
# time= linear spaced time series 
# N= standard gaussian noise
# A= nominal amperage range of mixer motor at 20 HP (sinusoidal) ~28A
#
# Components of simulated motor performance and decay:
#
#   S0= base inverse logrithmic carrier over the simulated motor life
#   S1= base sinusoidal operating current nominal 
#   FI1= fault injection - impulse at failure time 1
#   FI2= fault injection - impulse train at failure time 2
#   TC = total current:  S0 + S1 + S2 + N

numsamples = 100*100
time = np.linspace(1, numsamples, num=numsamples)
N = np.random.uniform(low=0.5, high=3, size=(numsamples,))  # noise
A = np.random.weibull(10., numsamples)  # amperage 
S0 = 1/np.log10(time+0.9)
S1 = 0.5 + 9* np.sin(time) 
FI1 = np.zeros_like(time)
FI2 = np.zeros_like(time)

print(time.shape)
print(N.shape)
print(A.shape)
print(S0.shape)
print(S1.shape)
print(FI1.shape)
print(FI2.shape)

print(time)
print(S0)

# build the data frame
df1 = pd.DataFrame()
df1['time'] = time
df1['N'] = N
df1['A'] = A
df1['S0'] = S0
df1['S1'] = S1
df1['FI1'] = FI1
df1['FI2]'] = FI2
df1['signal'] = S0 + S1 + FI1 + FI2 + N

print(df1.head(50))


print('writing output motor_data.csv')

filepath = Path('motor_data.csv')  
df1.to_csv(filepath, index=False)  
