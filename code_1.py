from math import inf
import numpy as np
import random
from scipy.stats import rankdata
from time import time

class Node:
    def __init__(self, solution, h):
        self.parent = None
        self.childrens = []
        self.h = h
        self.solution = solution
        self.hd = 0
    def geth(self):
        return self.h
    def getsolutionlength(self):
        return len(self.solution)
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


def fileRead(Fname, lenstrings):
    F = open(Fname, 'r')
    S = []
    for _ in range(lenstrings):
        S.append(F.readline().strip())
    return S

def hamming(s1, s2):
    i = 0
    counter = 0
    while(i < len(s1)):
        if(s1[i] != s2[i]):
            counter += 1
        i += 1
    return counter

def successor(node, S, al, lenstrings, level):
    node = node + al
    return node
def letterFrequency(S, al, lenstrings, level):
    x = 0
    for i in range(lenstrings):
        if S[i][level] == al:
            x += 1
    return x

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
def heuristicSimple(allnodes, S, level):
    # using global variable, not the best practice, but the alternative is to put everything inside the class...
    global hsim_per_string_cache
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
    return h

# this heuristic is as the one above (current prefix hamming similarity for each input string) but it also adds expected length of hamming similarity for suffix part
# the expected length is based on the most probable suffix solution formed by using the most frequent character in each character position (k-th character across all input strings)
def heuristicExpectedLength(allnodes, S, level, expectedSuffixesLengths):
    # using global variable, not the best practice, but the alternative is to put everything inside the class...
    global hsim_per_string_cache
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
            if len(hsim_per_string_cache)>0:
                hsim_j  = hsim_per_string_cache[j][candidate[:-1]]
            else:
                hsim_j = 0
            if candidate[level] == S[j][level]:
                hsim_j+=1
            new_hsim_per_string_cache[j][candidate] = hsim_j
            # calculating total expected length 
            if level<len(S[0])-1:
                hsim_j_whole = hsim_j+expectedSuffixesLengths[j][level+1]
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
    return h

def evaluation(allnodes, S, level, alphabet):
    l = len(allnodes)
    lens = len(S)
    lenalphabet = len(alphabet)
    h = []
    # for i in range(l):
    #     x = []
    #     for j in range(lens):
    #         # print(len(allnodes[i]), len(S[j][:level+1]))
    #         t = hamming(allnodes[i].getsolution(), S[j][:level+1])
    #         x.append(t)
    #     fn = max(x)
    #     t3 = 0
    #     # input(l)
    #     for z in range(lens):
    #         if x[z] == fn:
    #             t3 += 1
    #     t3 = (1 / lenalphabet) ** t3
    #     h_value = np.sum(x) + fn + 1 - t3
    #     h.append(1 / h_value)

    for i in range(l):
        x = 0
        for j in range(lens):
            if allnodes[i].getsolution()[level] == S[j][level]:
                x += 1
        h.append(x)
    return h

# when beam size is large, e.g. 10000 you should not use O(n^2) sorting!
def reduceBeamSlow(C,h,Beta):
    _rtn = []
    for i in range(Beta):
        t = -1
        ind = 0
        for j in range(len(h)):
            if h[j] > t:
                ind = j
                t = h[j]
        h[ind] = -1
        _rtn.append(C[ind])
    return _rtn

def reduceBeam(C,h,Beta):
    Ch = zip(C, h)
    ChSorted = sorted(Ch, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ChSorted][:Beta]

def randomGenerator(x):
    return random.uniform(0.0, float(x))

def expansion(node, al):
    no = Node(node.getsolution() + al, 0)
    no.setparent(node)
    return no

def getmainparent(node):
    if node.parent == None:
        return node
    return getmainparent(node.getparent())

def simulation(C, level, beta, alphabet, S):
    lenalphabet = len(alphabet)
    lencc = C[0].getsolutionlength()
    B = C
    lenB = len(B)
    counter = 0
    th = []
    for i in range(lenB):
        th.append(B[i].geth())
    tC = B
    while counter < level:
        if lencc + counter >= len(S[0]):
            break
        # rninter = (max(th) - min(th)) / 4# 8 or 16
        # rninter = 1.0
        lambda_ = 0.8
        lenth = len(th)
        # th = np.array(th) / max(th)
        for i in range(lenth):
            if th[i] == 0.0:
                th[i] = 1e-250
        thtemp = np.random.dirichlet(th, 1)
        for i in range(lenth):
            # rnd = randomGenerator(rninter)
            # input(thtemp[0][i])
            th[i] = ( lambda_ * th[i] ) + (( 1.0 - lambda_) * thtemp[0][i])
        tC = reduceBeam(tC, th, beta)
        childrens = []
        lentc = len(tC)
        for j in range(lentc):
            for i in range(lenalphabet):
                node = expansion(tC[j], alphabet[i])
                childrens.append(node)
        tC = childrens
        th = evaluation(tC, S, lencc + counter, alphabet)
        counter += 1
    lentc = len(tC)
    for i in range(lentc):
        x = 0
        for j in range(len(S)):
            t = hamming(tC[i].getsolution(), S[j][:lencc + counter])
            if t > x:
                x = t
        pnode = getmainparent(tC[i])
        if x > pnode.gethd():
            pnode.sethd(1 / x)
    return C

def startSimulation(nodes, h, gamma, level, beta, alphabet, S):
    newNodes = []
    for i in range(len(nodes)):
        newNodes.append(Node(nodes[i], h[i]))
    for i in range(gamma):
        newNodes = simulation(newNodes, level, beta, alphabet, S)
    th = []
    for i in range(len(newNodes)):
        th.append(newNodes[i].gethd())
    return th

def cunstructTable(S, alphabet):
    # S = ["acgtacgt", "atcggctt"]
    table = np.zeros((len(alphabet), len(S[0])))
    lens = len(S[0])
    for i in range(lens):
        for j in range(len(S)):
            for z in range(len(alphabet)):
                if S[j][i] == alphabet[z]:
                    table[z][i] += 1
    return rankassign(table)


def rankassign(table):
    table = rankdata(table, axis = 0)
    return table

def cunstructTable(S, alphabet):
    # S = ["acgtacgt", "atcggctt"]
    table = np.zeros((len(alphabet), len(S[0])))
    lens = len(S[0])
    for i in range(lens):
        for j in range(len(S)):
            for z in range(len(alphabet)):
                if S[j][i] == alphabet[z]:
                    table[z][i] += 1
    return rankassign(1 / table)

def correction1(expandedNodes, l, lfinal, alphabet, lenstrings, table, bestsolution, S):
    for i in range(l):
        ma = np.argmax(table1)
        for j in range(lfinal):
            # print(len(expandedNodes[j]), l)
            if expandedNodes[j][i] != alphabet[ma]:
                expandedNodes[j] = expandedNodes[j][:i] + alphabet[ma] + expandedNodes[j][i + 1:]
        tt = []
        for i in range(len(expandedNodes)):
            x = 0
            for j in range(lenstrings):
                t = hamming(expandedNodes[i], S[j])
                if t > x:
                    x = t
            tt.append(x)
        if max(tt) < bestsolution:
            bestsolution = max(tt)
            print("improved")
        table1 = np.sum(table, axis = 0) - table[:, :l]
    return bestsolution
def correction2(expandedNodes, l, lfinal, lenstrings, S):
    stepparameter = 20
    steps = int(l / stepparameter)
    final_solution = ""
    for i in range(steps):
        maxhamming = []
        for j in range(lfinal):
            x = 0
            for z in range(lenstrings):
                t = hamming(expandedNodes[j][i * stepparameter:(i + 1) * stepparameter], S[z][i * stepparameter:(i + 1) * stepparameter])
                if t > x:
                    x = t
            maxhamming.append(x)
        final_solution += expandedNodes[np.argmin(maxhamming)][i * stepparameter:(i + 1) * stepparameter]
    x = 0
    for j in range(lenstrings):
        t = hamming(final_solution, S[j])
        if t > x:
            x = t
    bestsolution = x
    return bestsolution

def correction3(expandednodes, S):
    lfinal = len(expandednodes)
    l = len(S[0])
    lenstrings = len(S)
    finalSolution = ""
    for i in range(l):
        ar = []
        for z in range(lfinal):
            x = 0
            for j in range(lenstrings):
                t = hamming(expandednodes[z][i:], S[j][i:])
                if t > x:
                    x = t
            ar.append(x)
        # print(np.argmax(ar))
        finalSolution += expandednodes[np.argmax(ar)][i]
    x = 0
    for i in range(lenstrings):
        t = hamming(finalSolution, S[i])
        if t > x:
            x = t
    return x

def caldistance(expandedNode, S):
    lenstrings = len(S)
    x = []
    for j in range(lenstrings):
        t = hamming(expandedNode, S[j])
        x.append(t)
    return x

def correction4(expandednodes, S, bestlength):
    l = len(expandednodes[0])
    for i in range(l):
        for j in range(len(expandednodes)):
            tempsolution = expandednodes[j]
            t = caldistance(expandednodes[j], S)
            ma = np.argmax(t)
            newtempsolution = ""
            for z in range(l):
                if i != ma:
                    newtempsolution += tempsolution[z]
                else:
                    newtempsolution += S[ma][z]
        tw = max(caldistance(newtempsolution, S))
        # input(tw)
        if  tw < bestlength:
            print("improvement happened")
            bestlength = tw
    return bestlength

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

def beamSearch(alphabet, S, g):
    lenstrings = len(S)
    childrens = []
    expandedNodes = [""]
    lenalphabet = len(alphabet)
    # maintable = cunstructTable(S, alphabet)
    # table = copy.deepcopy(maintable)
    # table1 = np.sum(table, axis = 0)
    lenl = len(S[0])
    level = 0
    beta = 10000 #20
    gamma = 150
    level1 = lenl / lenalphabet
    t_succ = 0
    t_heur = 0
    t_simu = 0
    t_redu = 0
    
    # used in expected length heuristic
    expectedSuffixesLengths = calculateExpectedSuffixesLengths(S)
    while level < lenl:
        lenexpand = len(expandedNodes)
        childrens = []
        start = time()
        for j in range(lenexpand):
            for i in range(lenalphabet):
                node = successor(expandedNodes[j], S, alphabet[i], lenstrings, level)
                childrens.append(node)
        t_succ+=(time()-start)
        start = time()
        #h = heuristicSimple(childrens, S, level)
        h = heuristicExpectedLength(childrens, S, level, expectedSuffixesLengths)
        best_h = max(h)
        t_heur+=(time()-start)
        start = time()
        #h = startSimulation(childrens, h, gamma, level1, 2, alphabet, S)
        t_simu+=(time()-start)
        if len(childrens) <= beta:
            expandedNodes = childrens
        else:
            start = time()
            expandedNodes = reduceBeam(childrens, h, beta)
            t_redu+=(time()-start)
        print("Level="+str(level)+ " best_h="+str(best_h)+" t_succ="+str(round(t_succ,1))+" t_heur="+str(round(t_heur,1))+" t_simu="+str(round(t_simu,1))+" t_redu="+str(round(t_redu,1)), end="\r")
        level += 1
    print("\n")
    best_i = -1
    best_d = inf
    for i in range(len(expandedNodes)):
        d = 0
        for j in range(lenstrings):
            t = hamming(expandedNodes[i], S[j])
            if t > d:
                d = t
        if d<best_d:
            best_d = d
            best_i = i
    best = expandedNodes[best_i]
    print("Hamming Distance Befor Correction: ", best_d)
    print(best)
    #l = len(expandedNodes[0])
    #lfinal = len(expandedNodes)
    # finalSolution = correction2(expandedNodes, l, lfinal, lenstrings, S)
    # finalSolution = correction4(expandedNodes, S, bestsolution)
    #finalSolution = correction2(expandedNodes, l, lfinal, lenstrings, S)
    #if finalSolution < bestsolution:
        # print("improved solution with correction")
    #    bestsolution = finalSolution
    #else:
    #    print("not improved correction")
    #print("Hamming Distance After Correction: ", bestsolution)


#os.chdir("C:/Users/Alireza/Documents/Research/Closest String Problem/Implementation")
S = fileRead("Tdata.txt", 10)
# alphabet = ['A', 'C', 'G', 'T']
alphabet = ['0', '1']
# alphabet = ['a', 'c', 'g', 't']
# alphabet = ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
beamSearch(alphabet, S, "")