from time import time
from pulp import LpProblem, lpSum, LpVariable, LpMinimize, LpBinary, LpInteger, PULP_CBC_CMD, LpStatus, value, LpSolution
import pulp as pl 
import orloge 
from os import remove
from math import ceil
from os import name
import os
import sys

# define cplex path
if name == 'nt': # windows
    cplex_path = r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex.exe'
else:
    cplex_path= r'/home/marko/Desktop/CPLEX_Studio127/cplex/bin/x86-64_linux/cplex'

def fileRead(Fname: str):
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

def alphabetDetection(S):
    alphabet = []
    for i in range(len(S)):
        for j in range(len(S[i])):
            if S[i][j] not in alphabet:
                alphabet.append(S[i][j])
    return alphabet

    
def conversion_chars_to_nums(strings: list):

    strings_converted = []
    global char_to_int
    
    for i in range(len(strings)):
        strings_converted.append( [] )
    
    char_num = 1
    
    for i, s in enumerate(strings):
        for si in s:
            
            if si in char_to_int.keys():
               strings_converted[i].append(char_to_int[si])
            else:
               strings_converted[i].append(char_num)
               char_to_int[si] = char_num
               char_num += 1 #next unused char
    
    return strings_converted
      
    
    
def ILP1(strings: list, alpha_size: int, timelimit: float, ind: int):


    K = alpha_size 
    
    model = LpProblem("CSP", LpMinimize)
    # define decision vars:
    z = LpVariable.dicts("z", [ (i, k) for i in range(len(strings))  for k in range(len(strings[0]))   ],  0, 1, LpBinary)
    d = LpVariable("d", 0, None)
    t = LpVariable.dicts("t", range(len(strings[0])), -1000000, None, LpInteger)
    
    # objective function (min d):
    model += d
    # constraints (2): 
    for i in range(len(strings)):
        model += lpSum([ z[ (i, k) ] for k in range(len(strings[0])) ] )  <= d
          
    ## constraints (3): 
    for i in range(len(strings)):
        for k in range(len(strings[0])):
            # print(strings[i][k])
            model += ( t[k] - strings[i][k] <= K * z[ (i, k) ] )
    
    ## constraints (4): 
    for i in range(len(strings)):
        for k in range(len(strings[0])):
            model += ( strings[i][k] -  t[k]  <= K * z[(i, k) ] )    
    
    

    # solve model:
    solver = pl.CPLEX_CMD(path=cplex_path, timeLimit=timelimit, 
             logPath="log_info"+str(ind)+".log", msg=False, threads=1)
    model.solve(solver) 
    # stats:
    solution = []
    for k in range(len(strings[0])):
        solution.append(int(t[k].varValue))
    # print("solution: ", solution)
    # print("Value: ", value(model.objective))

    # retrieve gap: 
    logs_dict = orloge.get_info_solver("log_info"+str(ind)+".log", "CPLEX" ) # Orloge returns a dict with all logs info
    best_bound, best_solution, status = logs_dict["best_bound"], logs_dict["best_solution"], logs_dict["status"]
    #print("Best bound: ", best_bound) 
    #print(logs_dict)
    remove("log_info"+str(ind)+".log")
    
    return best_solution, ceil(best_bound), int(status == "MIP - Integer optimal")


########### The third model from  Meneses et al. (2004): Section 2.4 
def ILP3(strings: list, alpha_size: int, timelimit: float, ind: int):

    # preprocessing:
    V = dict()
    for k in range(len(strings[0])):
        vk = set()
        for i in range(len(strings)):   
            vk.add(strings[i][k])
        
        V[k] = vk
        
    # building model:
    model3 = LpProblem("CSP-3", LpMinimize)
    # define decision vars V_k: k-th characters 
    v = LpVariable.dicts("v", [ (j, k) for j in  V[k] for k in range(len(strings[0]))   ],  0, 1, LpBinary)
    d = LpVariable("d", 0, None)
    
    # objective function (min d):
    model3 += d  
    
    # constraint (21): 
    for k in range(len(strings[0])):
        model3 += ( lpSum( [ v[ (j, k) ] for j in V[k] ] )  == 1)
    
    # constraint (22):   
    for i in range(len(strings)):  
        model3 += (len(strings[0]) - lpSum([  v[ ( strings[i][j], j) ]  for j in range(len(strings[0])) ] ) <= d )       
    
    # solve the model:
    solver = pl.CPLEX_CMD(path=cplex_path, timeLimit=timelimit, 
             logPath="log_info-3"+str(ind)+".log", msg=False, threads=1)
    model3.solve(solver)   #PULP_CBC_CMD(maxSeconds=timelimit, msg=True, fracGap=0))

    # stats -- retrieve gap: 
    logs_dict = orloge.get_info_solver("log_info-3"+str(ind)+".log", "CPLEX" ) # Orloge returns a dict with all logs info
    # print(logs_dict["progress"])
    best_bound, best_solution, status = logs_dict["best_bound"], logs_dict["best_solution"], logs_dict["status"]

    remove("log_info-3"+str(ind)+".log")
    
    return best_solution, ceil(best_bound), int(status == "MIP - Integer optimal") 

########## end of ILP-3 model implementation 


if __name__ == '__main__':

    # terminal parameters in the program
    # Calling procedure: python ILP.py <dtatse name> <maximum time limit> <number of ILP model (1 or 3)>

    # DEFAULT params and global vars:
    char_to_int = {}
    
    ### read terminal arguments:
    dataset = sys.argv[1]
    timelimit = int(sys.argv[2])
    model_num = int(sys.argv[3])
           
    strings, n_strings = fileRead(dataset)
    alphabet_length = len(alphabetDetection(strings))

    # preprocessing:
    strings_converted = conversion_chars_to_nums(strings) 

    ### Run ILP solver (CPLEX):
    print("ILP start to find solution, please wait...", end = '\r')
    curr = time()
    if model_num == 1:
        primal, dual, status = ILP1(strings_converted, alphabet_length, timelimit, 0)
    else:
        primal, dual, status = ILP3(strings_converted, alphabet_length, timelimit, 0)
            
    time_execute = time() - curr
    print("Solution: ", primal, " Dual bound: ",  dual, ' Time: ', time_execute)
