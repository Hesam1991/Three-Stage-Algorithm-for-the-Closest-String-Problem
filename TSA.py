from math import inf
import numpy as np
import random
from time import time
import os
import copy
import pandas as pd
import math
import cProfile
from statistics import mode
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "Datasets"

def fileRead(dataset, filename, lenstrings):
    file_path = DATASETS_DIR / dataset / filename

    with open(file_path, "r") as f:
        S = [f.readline().strip() for _ in range(lenstrings)]
        opt = f.readline().strip()

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




def reduceBeam1(C,h, dist, Beta):
    Ch = zip(C, dist, h)
    ChSorted = sorted(Ch, key=lambda x: x[2], reverse=True)
    return [x[0] for x in ChSorted][:Beta], [x[1] for x in ChSorted][:Beta]



def rankRecognize(alphabet, S, helper): 
    lenl = len(S[0])  
    rank1 = []
    rank2 = []
    if len(alphabet) == 2:
        for i in range(lenl):
            rank2.append(alphabet)
        return rank2
    for i in range(lenl):
        rank1.append(alphabetPruning(alphabet, i, S, 1))
        rank2.append(alphabetPruning(alphabet, i, S, 2))
    if helper == 1:
        return rank1
    else:
        return rank2

def newheuristicfunction(allnodes, probs, all_hamming_dist, level, S):
    h = []
    hm_dist = copy.deepcopy(all_hamming_dist)
    for i in range(len(hm_dist)):
        hm_dist[i] = np.array(hm_dist[i]) / max(hm_dist[i])
        var =  np.var(hm_dist[i])
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
    optimal_string = ""
    for i in range(len(s1)):
        if i % 2 == 0:
            optimal_string = optimal_string + s1[i]
        else:
            optimal_string = optimal_string + s2[i]
    return optimal_string


def find_critical_strings(base_string, S):
    hm_dists = overallhamming2(base_string, S)
    crit = max(hm_dists)
    critical_positions = []
    quasi_critical_positions = []
    for i in range(len(S)):
        if abs(crit - hm_dists[i]) <= 0:
            critical_positions.append(i)
        if abs(crit - hm_dists[i]) == 1:
            quasi_critical_positions.append(i)
    return critical_positions, quasi_critical_positions

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

    maximum_ = max(lst)
    newlst = []
    for i in range(len(lst)):
        if lst[i] == maximum_:
            newlst.append(i)
    indx = random.choice(newlst)

    return indx, letter_list[indx]


def localSearch3(base_string, S):
    current = ''.join(base_string)
    current_distance = max(overallhamming2(current, S))
    visited = {current}
    start_time = time()

    while time() - start_time < 5:
        critical_indices, _ = find_critical_strings(current, S)
        critical_strings = [S[i] for i in critical_indices]

        # Find the most frequent characters at each position.
        position_counts = []
        for j in range(len(current)):
            counts = {}
            for s in critical_strings:
                counts[s[j]] = counts.get(s[j], 0) + 1
            position_counts.append(counts)

        highest_frequency = max(max(counts.values()) for counts in position_counts)

        # Consider all tied position-character pairs, excluding unchanged moves.
        candidates = [(j, char) for j, counts in enumerate(position_counts) for char, count in counts.items() if count == highest_frequency and char != current[j]]
        random.shuffle(candidates)
        accepted = False

        for j, char in candidates:
            if time() - start_time >= 5:
                return list(current), current_distance

            candidate = current[:j] + char + current[j + 1:]
            if candidate in visited:
                continue

            candidate_distance = max(overallhamming2(candidate, S))
            if candidate_distance <= current_distance:
                current = candidate
                current_distance = candidate_distance
                visited.add(current)
                accepted = True
                break

        if not accepted:
            break

    return list(current), current_distance



def localSearch2(base_string, S):
    not_done = True
    base_string = list(base_string)
    start_time = time()
    best_hm_dist = np.max(overallhamming2(base_string, S))
    while time() - start_time <= 5:
        critical_strings_index, quasi_critical_positions = find_critical_strings(base_string, S)
        strings = []
        for item in critical_strings_index:
            strings.append(S[item])
        quasi_strings = []
        for item in quasi_critical_positions:
            quasi_strings.append(S[item])
        index, letter = findPositionWithSameLetter(strings)
        temp_string = copy.deepcopy(base_string)
        temp_string[index] = letter

        hm_check_strings = [s[index] for s in strings]
        hm_check_quasi_strings = [s[index] for s in quasi_strings]
        x = 0
        flag = True
        for ch_s in hm_check_strings:
            if letter != ch_s:
                x = best_hm_dist + 1
                flag = False
                break
        if flag == True:
            for ch_s in hm_check_quasi_strings:
                if letter != ch_s:
                    x = best_hm_dist
                    flag = False
                    break
        if flag == True:
            if hm_check_strings.count(letter) == len(hm_check_strings):
                x = best_hm_dist - 1
                base_string = temp_string
                # break
        if flag == True:
            x = best_hm_dist
        if random.random() > 0.5:
            base_string = temp_string
    return base_string, best_hm_dist

def localSearch(base_string, S):
    not_done = True
    base_string = list(base_string)
    start_time = time()
    tr = 0
    while time() - start_time <= 5:
        x = np.max(overallhamming2(base_string, S))
        critical_strings_index, _ = find_critical_strings(base_string, S)
        strings = []
        for item in critical_strings_index:
            strings.append(S[item])
        index, letter = findPositionWithSameLetter(strings)
        temp_string = copy.deepcopy(base_string)
        temp_string[index] = letter
        temp_x = np.max(overallhamming2(temp_string, S))
        if temp_x <= x:
            base_string = temp_string
            # if temp_x < x: #first improvement strategy
            #     return base_string, temp_x
    return base_string, max(overallhamming2(base_string, S))


# def betaSpecification(beta, t_rem, t_rem_prime):
#     if t_rem_prime == 0:
#         t_rem_prime = 1
#     if t_rem / t_rem_prime > 1.1:
#         return math.ceil(beta*1.1)
#     elif t_rem / t_rem_prime < 0.9:
#         return min(math.ceil(beta / 1.1), 150)
#     return beta

def betaSpecification(beta, t_rem, t_rem_prime):
    if t_rem_prime == 0:
        t_rem_prime = 1
    if t_rem / t_rem_prime > 1.1:
        return math.ceil(beta*1.1)
    elif t_rem / t_rem_prime < 0.9:
        return min(math.ceil(beta / 1.1), 150)
    return beta

def beamSearch(alphabet, S, beta, helper, trail_phase, TRBS):
    lenstrings = len(S)
    childrens = []
    expandedNodes = [""]
    lenalphabet = len(alphabet)
    lenl = len(S[0])
    level = 0
    hsim_per_string_cache = []
    rankedalphabet = rankRecognize(alphabet, S, helper)
    all_hamming_dist = [[0 for i in range(lenstrings)]]
    expectedSuffixesLengths = calculateExpectedSuffixesLengths(S)
    t_totall = time()
    h = []
    beat_total_time = time()
    if lenl < 400:
        t_max = 30.0
    elif lenl >= 400 and lenl < 1000:
        t_max = 60.0
    else:
        t_max = 120.0
    while level < lenl:
        last_iter_time_start = time()
        lenexpand = len(expandedNodes)
        childrens = []
        alphabet_succ =  rankedalphabet[level]
        lenalphabet_succ = len(alphabet_succ)
        temp_h_dist = []
        for j in range(lenexpand):
            for i in range(lenalphabet_succ):
                node = successor(expandedNodes[j], S, alphabet_succ[i], lenstrings, level)
                temp_h_dist.append(np.add(all_hamming_dist[j], overallhamming1(alphabet_succ[i], S, level)).tolist())
                childrens.append(node)
        all_hamming_dist = temp_h_dist
        h, hsim_per_string_cache = heuristicExpectedLength(childrens, S, level, expectedSuffixesLengths, hsim_per_string_cache)
        mo = mode(h)
        if h.count(mo) == len(h) and len(h) != 1:
            h = newheuristicfunction(childrens, 0, all_hamming_dist, level, S)
        if len(childrens) <= beta:
            expandedNodes = childrens
        else:
            expandedNodes, all_hamming_dist = reduceBeam1(childrens, h, all_hamming_dist, beta)
        if trail_phase == False and TRBS == True:
            last_iter_time = time() - last_iter_time_start
            t_rem = t_max - (time() - beat_total_time)
            t_rem_prime = (lenl - level) * last_iter_time
            beta = betaSpecification(beta, t_rem, t_rem_prime)
            # print(t_rem, t_rem_prime)
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
    final_solution = expandedNodes[best_candidate]
    if trail_phase == False:
        new_solution, local_dist = localSearch3(expandedNodes[best_candidate], S)
        if local_dist < best_hm:
            best_hm = local_dist
            final_solution = new_solution
            print('Improvement done!')
    return best_hm, (time() - t_totall), final_solution



trail_beta = 15
beta = 300
trail_TRBS = False
main_TRBS = True

alphabets = {
    "Alpha-4": ['A', 'C', 'G', 'T'],
    "Alpha-20": ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
}

numStringsList = [10, 15, 20, 25, 40, 60, 80, 100]
stringLengthList = [50, 100, 150, 200, 250, 300, 350, 400, 800, 1000, 1500]

for dataset, alphabet in alphabets.items():
    for n in numStringsList:
        for L in stringLengthList:

            fname = f"{len(alphabet)}_{n}_{L}"
            S, _ = fileRead(dataset, fname, n)

            D1, time_trial_1, _ = beamSearch(alphabet, S, trail_beta, helper=1, trail_phase=True, TRBS=trail_TRBS)
            D2, time_trial_2, _ = beamSearch(alphabet, S, trail_beta, helper=2, trail_phase=True, TRBS=trail_TRBS)

            if D1 < D2:
                rank = 1
            elif D2 < D1:
                rank = 2
            else:
                rank = random.choice([1, 2])

            D, main_time, solution = beamSearch(alphabet, S, beta, helper=rank, trail_phase=False, TRBS=main_TRBS)

            total_time = time_trial_1 + time_trial_2 + main_time

            print(dataset, fname, 'solution:', D, 'time:', total_time)