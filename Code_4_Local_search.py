from math import inf
import numpy as np
import pandas as pd
import random
from scipy.stats import rankdata
from time import time
import os
import copy
from sklearn.preprocessing import normalize
from scipy.spatial.distance import hamming as hmdistance
import pandas as pd
import math
from scipy import stats
import multiprocessing
from collections import Counter
import cProfile
from math import floor
from scipy.special import softmax
from statistics import mode
from joblib import dump, load
from scipy.stats import rankdata


class Node:
    def __init__(self, solution, h, lst):
        self.parent = None
        self.childrens = []
        self.h = h
        self.solution = solution
        self.hd = 0
        self.leaf = True
        self.N = 1
        self.hamming = lst
        self.maxh = 1
        self.maxn = 1
        self.childrenh = []
        self.childrenn = []
    def geth(self):
        return self.h
    def seth(self, obj):
        self.h = obj
    def getleaf(self):
        return self.leaf
    def setN(self):
        self.N += 1
    def getN(self):
        return self.N
    def setleaf(self, obj):
        self.leaf = obj
    def getsolution(self):
        return self.solution
    def setparent(self, parentnode):
        self.parent = parentnode
    def getparent(self):
        return self.parent
    def gethd(self):
        return self.hd
    def sethd(self, obj):
        self.hd = obj
    def setchildern(self, obj):
        self.childrens.append(obj)
    def gethn(self):
        return self.childrenh, self.childrenn
    def getmaxhn(self):
        return self.maxh, self.maxn
    def getchildren(self):
        return self.childrens
    def gethamming(self):
        return self.hamming


def fileRead(Fname, lenstrings):
    F = open(Fname, 'r')
    S = []
    for _ in range(lenstrings):
        S.append(F.readline().strip())
    opt = F.readline().strip()
    return S, opt

def successor(node, S, al, lenstrings, level):
    node = node + al
    return node

def calculateExpectedSuffixesLengths(S):
    SSize = len(S[0])
    Scount = len(S)
    # this is used in the expected length heuristic
    expectedSolution = []
    for i in range(SSize):
        # looking at the most frequent character at position i
        freq = {}
        for j in range(Scount):
            c = S[j][i]
            if c in freq:
                freq[c]+=1
            else:
                freq[c]=1
        # sort freq by value
        mfc = sorted(freq.items(), key=lambda item: item[1], reverse=True)[0][0]
        expectedSolution.append(mfc)
    # now constructing expected length for each suffix of each input string based on the expected solution string
    expectedSuffixesLengths = []
    for j in range(Scount):
        jLengths = []
        currLen = 0
        # doing it backwards
        for i in reversed(range(SSize)):
            if expectedSolution[i]==S[j][i]:
                currLen+=1
            jLengths.append(currLen)
        jLengths = list(reversed(jLengths))
        expectedSuffixesLengths.append(jLengths)
    return expectedSuffixesLengths

def heuristicSimpleSlow(allnodes, S, level):
    l = len(allnodes)
    lens = len(S)
    h = []
    for i in range(l):
        candidate = allnodes[i]
        min_hamming_sim = inf
        for j in range(lens):
            hamming_sim = 0
            for k in range(level+1):
                if candidate[k] == S[j][k]:
                    hamming_sim+=1
            if hamming_sim<min_hamming_sim:
                min_hamming_sim = hamming_sim
        h.append(min_hamming_sim)
    return h

# used to cache hamming similarities w.r.t. each of M input strings
# there are M dictionaries inside this array, where j-th one contains mapping from candidate string to its hamming similarity with j-th input string
hsim_per_string_cache = []
# this heuristics is the same as the actual objective, just looking at the minimal similarity of candidate string w.r.t. partial input strings
def heuristicSimple(allnodes, S, level, hsim_per_string_cache):
    # using global variable, not the best practice, but the alternative is to put everything inside the class...
    l = len(allnodes)
    lens = len(S)
    h = []
    # this is faster one, using previously cached values
    new_hsim_per_string_cache = [dict() for x in range(lens)]
    for i in range(l):
        candidate = allnodes[i]
        min_hsim = inf
        for j in range(lens):
            if len(hsim_per_string_cache)>0:
                hsim_j  = hsim_per_string_cache[j][candidate[:-1]]
            else:
                hsim_j = 0
            if candidate[level] == S[j][level]:
                hsim_j+=1
            new_hsim_per_string_cache[j][candidate] = hsim_j
            if hsim_j<min_hsim:
                min_hsim = hsim_j
        h.append(min_hsim)
    hsim_per_string_cache = new_hsim_per_string_cache
    return h, hsim_per_string_cache

# this heuristic is as the one above (current prefix hamming similarity for each input string) but it also adds expected length of hamming similarity for suffix part
# the expected length is based on the most probable suffix solution formed by using the most frequent character in each character position (k-th character across all input strings)
def heuristicExpectedLength(allnodes, S, level, expectedSuffixesLengths, hsim_per_string_cache):
    # using global variable, not the best practice, but the alternative is to put everything inside the class...
    # global hsim_per_string_cache
    # print('\n', hsim_per_string_cache)
    l = len(allnodes)
    lens = len(S)
    h = []
    # this is faster one, using previously cached values
    new_hsim_per_string_cache = [dict() for x in range(lens)]
    max_min_hsim = 0

    for i in range(l):
        candidate = allnodes[i]
        min_hsim = inf
        for j in range(lens):
            if len(hsim_per_string_cache) > 0:
                hsim_j  = hsim_per_string_cache[j].get(candidate[:-1])
                if hsim_j == None:
                    hsim_j = 0
            else:
                hsim_j = 0
            if candidate[level] == S[j][level]:
                hsim_j+=1
            new_hsim_per_string_cache[j][candidate] = hsim_j
            # calculating total expected length
            if level<len(S[0])-1:
                hsim_j_whole = hsim_j + expectedSuffixesLengths[j][level+1]
            else:
                hsim_j_whole = hsim_j
            if hsim_j_whole<min_hsim:
                min_hsim = hsim_j_whole
            # early exit if the min_hsim is already too bad
            if min_hsim<max_min_hsim-3:
                min_hsim = 0
                break
        # remembering the global best
        if min_hsim>max_min_hsim:
            max_min_hsim = min_hsim
        h.append(min_hsim)
    hsim_per_string_cache = new_hsim_per_string_cache
    return h, new_hsim_per_string_cache

def alphabetPruning(alphabet, ii, Strings, rank_score):
    _rtn_alphabet = []
    stprop = dict()
    for ch in alphabet:
        x = 0
        for st in Strings:
            x += st[ii:ii+1].count(ch)
        stprop[ch] = x
    for i in range(rank_score):
        for key, value in stprop.items():
            if max(stprop.values()) == value and key not in _rtn_alphabet:
                _rtn_alphabet.append(key)
        for k in _rtn_alphabet:
          stprop[k] = -1
    return _rtn_alphabet

def overallhamming2(s1, S):
    dists = []
    for ss in S:
        counter = 0
        lens = len(s1)
        for i in range(lens):
            if s1[i] != ss[i]:
                counter += 1
        dists.append(counter)
    return dists

def overallhamming1(s1, S, level):
    lst = []
    for i in range(len(S)):
        if S[i][level] == s1:
            lst.append(0)
        else:
            lst.append(1)
    return lst

def overallhamming(s1, S):
    dists = []
    for ss in S:
        counter = 0
        lens = len(ss)
        for i in range(lens):
            if s1[i] != ss[i]:
                counter += 1
        dists.append(counter)
    return max(dists)

def hamming_2_string(s1, s2):
    hm = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            hm += 1
    return hm
def reduceBeam(C,h, Beta):
    Ch = zip(C, h)
    ChSorted = sorted(Ch, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ChSorted][:1]



def reduceBeam1(C,h, dist, Beta):
    Ch = zip(C, dist, h)
    ChSorted = sorted(Ch, key=lambda x: x[2], reverse=True)
    return [x[0] for x in ChSorted][:Beta], [x[1] for x in ChSorted][:Beta]


def getmainparent(node):
    if node.parent == None:
        return node
    return getmainparent(node.getparent())

def expansion(node, al, level, S):
    x = node.getsolution() + al
    lst = node.gethamming()
    hmd = overallhamming1(al, S, level)
    lst = np.add(lst, hmd)
    no = Node(x, 1, lst)
    no.setparent(node)
    return no, x

def simulation(C, currentlevel, alphabet, S, expectedSuffixesLengths, lambda_, lambda_prime, step, rankedalphabet):
    tC = C
    hsim_per_string_cache = []
    counter = 0
    while True:
        tN = [i.getN() for i in tC]
        th = [i.geth() for i in tC]
        tN = np.divide(tN, max(tN))
        th = np.divide(th, max(th))
        th[th <= 0] = 1e-25
        thtemp = np.random.dirichlet((1 / tN) + th, 1)[0]
        th = np.add(lambda_ * th, lambda_prime * thtemp)
        tC = reduceBeam(tC, th, 1)[0]
        tC.setN()
        if tC.getleaf():
            break
        tC = tC.getchildren()
        counter += 1
    level = currentlevel + counter - 1
    if level + 1 >= len(S[0]):
        return C
    tC.setleaf(False)
    alphabet = rankedalphabet[level + 1]
    hdPlus = []
    for ch in alphabet:
        node, sol = expansion(tC, ch, level + 1, S)
        node.seth(np.median(node.gethamming()))
        tC.setchildern(node)
        hdPlus.append(node.gethamming())
    hd = np.mean(hdPlus)

    mparent = getmainparent(node)
    if ((1/hd) / (level + 1)) > mparent.gethd():
        mparent.sethd((1/hd) / (level + 1))
    return C



def startSimulation(nodes, h, gamma, level, alphabet, S, expectedSuffixesLengths, lambda_, rankedalphabet, all_hamming_dist):
    newNodes = [Node(node, hd, lst) for node, hd, lst in zip(nodes, h, all_hamming_dist)]
    for step in range(gamma):
        newNodes = simulation(newNodes, level, alphabet, S, expectedSuffixesLengths, lambda_[step], 1.0 - lambda_[step], step, rankedalphabet)
    th = [node.gethd() for node in newNodes]
    return th

def rankRecognize(alphabet, S): 
    lenl = len(S[0])  
    rank1 = []
    rank2 = []
    rank3 = []
    if len(alphabet) == 2:
        for i in range(lenl):
            rank2.append(alphabet)
        return rank2
    for i in range(lenl):
        rank1.append(alphabetPruning(alphabet, i, S, 1))
        rank2.append(alphabetPruning(alphabet, i, S, 2))
        rank3.append(alphabetPruning(alphabet, i, S, 3))
    lens1, lens2 = [], []
    for i in range(lenl):
        lens1.append(len(rank1[i]))
        lens2.append(len(rank2[i]))
    r1, r2, s1, s2 = np.mean(lens1), np.mean(lens2), np.std(lens1), np.std(lens2)
    # print('Manual ranking, fix it.')
    # return (rank2)
    if len(alphabet) == 4 and len(S) <= 10:
        return rank1
    if r1 < 1.2:
        return rank2
    else:
        if r2 >= 4.0:
            return rank1
        else:
            return rank2

def newheuristicfunction(allnodes, probs, all_hamming_dist, level, S):
    h = []
    for i in range(len(all_hamming_dist)):
        var =  np.var(all_hamming_dist[i])
        h.append(var + 1)
    return 1 / np.array(h)



#upper bound for two strings
def upperbound(s1, s2):
    mismatches = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            mismatches += 1
    return math.ceil(mismatches / 2)

def exact2string(s1, s2):
    hm = math.ceil(hamming_2_string(s1, s2) / 2)
    print(hm)
    optimal_string = ""
    for i in range(len(s1)):
        if i % 2 == 0:
            optimal_string = optimal_string + s1[i]
        else:
            optimal_string = optimal_string + s2[i]
    return optimal_string

def ubH(SS, probs):
    S = copy.deepcopy(SS)
    while len(S) != 1:
        newS = []
        for i in range(len(S) - 1):
            x = exact2string(S[i], S[i+1])
            newS.append(x)
        S = newS
    return S[0]

def calculateLetterFreq(inputStrings, alphabet):
    stringLength = len(inputStrings[0])
    letterFreqTable = {}
    for position in range(stringLength):
        alphabetFreqTable = {}
        for alphabetLetter in alphabet:
            alphabetFreqTable[alphabetLetter] = 0
            for inputString in inputStrings:
                if inputString[position] == alphabetLetter:
                    alphabetFreqTable[alphabetLetter] += 1
        letterFreqTable[position] = alphabetFreqTable
    return letterFreqTable

def eq_check(S, indexes, position):
    flag = True
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            if S[indexes[i]][position] != S[indexes[j]][position]:
                flag = False
        return flag



def rankassign(lst):
    h = [inf for _ in range(len(lst))]
    rank = 0
    while inf in h:
        ma = min(lst)
        for i in range(len(lst)):
            if lst[i] == ma:
                h[i] = rank
                lst[i] = inf
        rank += 1
    return max(h) - np.array(h)


'''def find_most_freq_positions(news, alphabet, freq_table):
    freq_score_boared = []
    lenl = len(news[0])
    max_val = 0
    ind_max = 0
    ch_max = ''
    for i in range(lenl):
        if max_val < max(freq_table[i].values()):
            max_val = max(freq_table[i].values())
            ind_max = i
            ch_max = max(zip(freq_table[i].values(), freq_table[i].keys()))[1]
    return ind_max, ch_max
def localSearch(base_string, S, alphabet):
    overall_hm = overallhamming2(base_string, S)
    base_distance = max(overall_hm)
    improvement = False
    start_time = time()
    base_string = list(base_string)
    best_string = list(base_string)
    while not improvement and (time() - start_time) <= 5:
        cr_str = find_critical_strings(base_string, S)
        news = []
        for item in cr_str:
            news.append(S[item])
        freq_table = calculateLetterFreq(news, alphabet)
        ind, ch = find_most_freq_positions(news, alphabet, freq_table)
        base_string[ind] = ch
        x = overallhamming2(base_string, S)
        if max(x) < base_distance:
            base_distance = max(x)
            best_string = copy.deepcopy(base_string)
    return max(overallhamming2(best_string, S))'''

def find_critical_strings(base_string, S):
    hm_dists = overallhamming2(base_string, S)
    crit = max(hm_dists)
    critical_positions = []
    for i in range(len(S)):
        if abs(crit - hm_dists[i]) <= 0:
            critical_positions.append(i)
    return critical_positions

def most_frequent(List):
    return max(set(List), key = List.count)

def findPositionWithSameLetter(strings):
    strings = np.array(strings)
    lenl = len(strings[0])
    lst = []
    letter_list = []
    for i in range(lenl):
        c_list = [strings[j][i] for j in range(len(strings))]
        letter = most_frequent(c_list)
        lst.append(c_list.count(letter))
        letter_list.append(letter)
    return np.argmax(lst), letter_list[np.argmax(lst)]

def localSearch(base_string, S):
    not_done = True
    base_string = list(base_string)
    start_time = time()
    best_hm_dist = np.max(overallhamming2(base_string, S))
    while time() - start_time <= 5:
        critical_strings_index = find_critical_strings(base_string, S)
        strings = []
        for item in critical_strings_index:
            strings.append(S[item])
        index, letter = findPositionWithSameLetter(strings)
        temp_string = copy.deepcopy(base_string)
        temp_string[index] = letter
        if np.max(overallhamming2(temp_string, S)) <= best_hm_dist:
            base_string = temp_string
            best_hm_dist = np.max(overallhamming2(base_string, S))

    return base_string, max(overallhamming2(base_string, S))

def graspHeuristic(allnodes, S, all_hamming_dist, lenalphabet):
    h = []
    lenl = len(S[0])
    len_partial = len(allnodes[0])
    len_candidates = len(all_hamming_dist)
    if len_partial < lenl:
        for i in range(len_candidates):
            f_x = max(all_hamming_dist[i])
            cost = f_x + 1 - ( (1/lenalphabet) ** all_hamming_dist[i].count(f_x))
            h.append(cost)
    else:
        for i in range(len_candidates):
            f_x = max(all_hamming_dist[i])
            cost = f_x + 1 - ( ( (lenalphabet - 1) /lenalphabet) ** all_hamming_dist[i].count(f_x)) - ( ( (1/lenalphabet) ** all_hamming_dist[i].count(f_x) ) * ( ( (lenalphabet - 1) /lenalphabet) ** all_hamming_dist[i].count(f_x - 1)) )
            h.append(cost)
    return np.array(h)

def beamSearch(alphabet, S, g, helper = 0):
    lenstrings = len(S)
    childrens = []
    expandedNodes = [""]
    lenalphabet = len(alphabet)
    lenl = len(S[0])
    level = 0
    beta = 100
    gamma = 100
    # print("parameters: ", "beta: ", beta, " gamma: ", gamma, " depth: ", depth, " secondbeta: ", secondbeta)
    t_succ = 0
    t_heur = 0
    t_simu = 0
    t_redu = 0
    if lenl < 2000:
        a = 0.0039
        b = 0.3092 
    else:
        a = 0.002
        b = 0.7
    lambda_ = [min((a * step) + b, 1) for step in range(gamma, 2*gamma)]
    hsim_per_string_cache, hsim_per_string_cache2 = [], []
    rankedalphabet = rankRecognize(alphabet, S)
    all_hamming_dist = [[0 for i in range(lenstrings)]]
    # used in expected length heuristic
    expectedSuffixesLengths = calculateExpectedSuffixesLengths(S)
    t_totall = time()
    h = []
    # freq_table = calculateLetterFreq(S, alphabet)
    # probs = []
    # for i in range(lenstrings):
    #     st_prob = 1.0
    #     for j in range(lenl):
    #         st_prob = st_prob * freq_table[j][S[i][j]]
    #     probs.append(st_prob)
    while level < lenl:
        lenexpand = len(expandedNodes)
        childrens = []
        start = time()
        alphabet_succ =  rankedalphabet[level]
        lenalphabet_succ = len(alphabet_succ)
        temp_h_dist = []
        temp_strings = [s[:level+1] for s in S]
        for j in range(lenexpand):
            for i in range(lenalphabet_succ):
                node = successor(expandedNodes[j], S, alphabet_succ[i], lenstrings, level)
                temp_h_dist.append(np.add(all_hamming_dist[j], overallhamming1(alphabet_succ[i], S, level)).tolist())
                childrens.append(node)
        all_hamming_dist = temp_h_dist
        h, hsim_per_string_cache = heuristicExpectedLength(childrens, S, level, expectedSuffixesLengths, hsim_per_string_cache)
        mo = mode(h)
        sim = True
        if h.count(mo) == len(h) and len(h) != 1:
            sim = False 
            h = newheuristicfunction(childrens, 0, all_hamming_dist, level, S)
        best_h = max(h)
        if sim:
            h = startSimulation(childrens, h, gamma, level + 1, alphabet, S, expectedSuffixesLengths, lambda_, rankedalphabet, all_hamming_dist)

        if len(childrens) <= beta:
            expandedNodes = childrens
        else:
            expandedNodes, all_hamming_dist = reduceBeam1(childrens, h, all_hamming_dist, beta)
        level += 1

    best_hm = inf
    best_candidate = 0
    j = 0
    for i in expandedNodes:
      hm = overallhamming(i, S)
      if hm < best_hm:
        best_hm = hm 
        best_candidate = j
      j += 1

    # local_answer, local_dist = graspLocalSearch(expandedNodes[best_candidate], rankedalphabet, S)
    # st_time = time()
    new_solution, local_dist = localSearch(expandedNodes[best_candidate], S)
    # print('time for improvement: ', time() - st_time)
    if local_dist < best_hm:
        best_hm = local_dist
        print('improvment done!')

    return best_hm, (time() - t_totall)




# os.chdir("C:/Users/Alireza/Documents/Research/Closest String Problem/Implementation/test cases/McClure")
# McClureDataset = ['McClure-582-20-6-141',
#                   'McClure-582-20-10-141',
#                   'McClure-582-20-12-141',
#                   'McClure-586-20-6-100',
#                   'McClure-586-20-10-98',
#                   'McClure-586-20-12-98'
# ]
# numStringsList = [6, 10, 12, 6, 10, 12]
# McClureAlphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# max_iter = 10
# for data, numstr in zip(McClureDataset, numStringsList):
#     m = []
#     time_list = []
#     for i in range(max_iter):
#         start = time()
#         S, _ = fileRead(data, numstr)
#         x, total_time = beamSearch(McClureAlphabet, S, "", 0)
#         m.append(x)
#         time_list.append(total_time)
#     print(data, 'best: ', min(m), ' worst: ', max(m), ' average: ', np.mean(m), np.mean(time_list), '\n')
# input()





# ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
p53_protein_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
os.chdir("C:/Users/Alireza/Documents/Research/Closest String Problem/Implementation/test cases/random")
alphabet = [
    ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
]
# numStringsList = [5]
# stringLengthList = [362, 381, 285]
numStringsList = [40, 60, 80, 100]
stringLengthList = [50, 100, 150, 200, 250, 300, 350, 400]
# stringLengthList = [800, 1000, 1500]
for z in range(len(alphabet)):
    for i in range(len(numStringsList)):
        for j in range(len(stringLengthList)):
            fname = str(len(alphabet[z])) + "_" + str(numStringsList[i]) + "_" + str(stringLengthList[j])
            # fname = 'TP53_' + str(numStringsList[i]) + '_' + str(stringLengthList[j])
            # fname = 'P53_' + str(numStringsList[i]) + '_' + str(j+1) + '_' + str(stringLengthList[j])
            S, _ = fileRead(fname, numStringsList[i])
            max_iter = 10
            m = []
            time_list = []
            for _ in range(max_iter):
                D, total_time = beamSearch(alphabet[z], S, "", helper = 1)
                m.append(D)
                time_list.append(total_time)
            print(fname, 'best: ', min(m), ' worst: ', max(m), ' average: ', np.mean(m), ' time: ', np.mean(time_list))