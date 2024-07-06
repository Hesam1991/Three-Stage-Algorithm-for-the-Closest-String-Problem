import numpy as np
import random
import math
import copy
from time import time
import os
import sys


def fileRead(Fname):
    f = open(Fname, 'r')
    strings = []
    n = 0
    while(True): 
        line = f.readline()
        if line == "":
            break
        else:
            n = n + 1
            s = line.strip()
            strings.append(s)
    return strings, n
def alphabetDetection(S: list):
    alphabet = []
    for i in range(len(S)):
        for j in range(len(S[i])):
            if S[i][j] not in alphabet:
                alphabet.append(S[i][j])
    return alphabet
def firstChSelection(strings: list, alphabet: list):
    lst = []
    for j in range(len(alphabet)):
        x = 0
        for i in range(len(strings)):
            if strings[i][0] == alphabet[j]:
                x += 1
        lst.append(x)
    return alphabet[np.argmax(lst)]

def overallhamming(s1: str, S: list):
    dists = []
    for ss in S:
        counter = 0
        lens = len(s1)
        for i in range(lens):
            if s1[i] != ss[i]:
                counter += 1
        dists.append(counter)
    return dists

def sstr(strings: list, bound: int):
    newstrings = []
    for i in range(len(strings)):
        newstrings.append(strings[i][0:bound])
    return newstrings

def GW(strings: list, alphabet: list):
    lenl = len(strings[0])
    solution = firstChSelection(strings, alphabet)
    for i in range(1, lenl):
        slicedstrings = sstr(strings, i)
        scores = overallhamming(solution, slicedstrings)
        solution = solution + strings[np.argmax(scores)][i]
    return solution

def findfreq(strings: list, alphabet: list):
    freq_table = []
    for j in range(len(strings[0])):
        selected_char = []
        for i in range(len(alphabet)):
            x = 0
            for k in range(len(strings)):
                if strings[k][j] == alphabet[i]:
                    x += 1
            selected_char.append(x)
        freq_table.append(alphabet[np.argmax(selected_char)])
    return freq_table


def mutate(t: str, alphabet: list, change_rate: float, strings: list, freq_table: list):
    lenl = len(strings[0])
    positions = []
    i, j = 0, 0
    temp_t = copy.deepcopy(t)
    flag = False
    while True:
        if i >= math.floor(lenl * change_rate):
            break
        if len(positions) >= lenl - 1:
            flag = True
            break
        x = random.randint(0, lenl - 1)
        if x not in positions:
            positions.append(x)
        if temp_t[x] != freq_table[x]:
            temp_t = temp_t[:x] + freq_table[x] + temp_t[x + 1:]
            i += 1
        j += 1
    if flag == True:
        ind = np.argmax(overallhamming(temp_t, strings))
        while i < math.floor(lenl * change_rate):
            x = random.randint(0, lenl - 1)
            temp_t = temp_t[:x] + strings[ind][x] + temp_t[x + 1:]
            i += 1
    return temp_t

def SA(init_solution: str, strings: list, alphabet: list, change_rate:float, T_max: int, L: int, T_condition: int, gamma: float, freq_table: list):
    t = init_solution
    T = T_max
    for KK in range(T_condition):
        for i in range(L):
            t_prime = mutate(t, alphabet, change_rate, strings, freq_table)
            delta = max(overallhamming(t_prime, strings)) - max(overallhamming(t, strings))
            if delta < 0 or (delta > 0 and ((math.e ** (-delta / T)) > random.random())):
                t = t_prime
        T = gamma * T
    return t

if __name__ == '__main__':
    
    # Calling procedure: python GWSA.py <dtatse name>
    #Initial parameters
    change_rate, T_max, L, T_condition, gamma = 0.04, 5, 500, 10, 0.8 

    #Inputs
    dataset = sys.argv[1]

    #File read and identify alphabets
    S, n_strings = fileRead(dataset) 
    alphabet = alphabetDetection(S)

    #preprocess
    freq_table = findfreq(S, alphabet)

    print("GWSA Starts to find CS, please wait...", end = '\r')
    #Start GWSA
    start_time = time()
    init_solution = GW(S, alphabet)
    final_solution = SA(init_solution, S, alphabet, change_rate, T_max, L, T_condition, gamma, freq_table)
    dist = max(overallhamming(final_solution, S))
    ovarall_time = time() - start_time
    
    #print CS, distance, and time
    print("Closest String: ", final_solution)
    print("Distance: ", dist)
    print("Running Time: ", ovarall_time)