from time import time
from pulp import LpProblem, lpSum, LpVariable, LpMinimize, LpBinary, LpInteger, PULP_CBC_CMD, LpStatus, value, LpSolution
import pulp as pl 
import orloge # pip install orloge
from os import remove
from math import ceil
from os import name
import sys, getopt

# define cplex path
if name == 'nt': # windows
    cplex_path = r'C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex.exe'
else:
    cplex_path= r'/home/marko/Desktop/CPLEX_Studio127/cplex/bin/x86-64_linux/cplex'


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

def read_instance(path):


     strings = []
     f = open(path, "r")
     params = f.readline()
     
     m, alpha = params.split("\t")
     m, alpha = int(m.strip()), int(alpha.strip())
     
     print(m, " ---- ", alpha)
     
     while(True): 
         
         line = f.readline()
         if line == "":
             break
         else:
             
             n, string = line.split("\t")
             s = string.strip()
             strings.append(s)
     
     return strings, alpha
      
    
    
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
            model += ( t[k] - strings[i][k] <= K * z[ (i, k) ] )
    
    ## constraints (4): 
    for i in range(len(strings)):
        for k in range(len(strings[0])):
            model += ( strings[i][k] -  t[k]  <= K * z[(i, k) ] )    
    

    # solve model:
    solver = pl.CPLEX_CMD(path=cplex_path, timeLimit=timelimit, 
             logPath="log_info"+str(ind)+".log", msg=False, threads=1)
    model.solve(solver)   #PULP_CBC_CMD(maxSeconds=timelimit, msg=True, fracGap=0))
    # stats:
    
    solution = []
    for k in range(len(strings[0])):
        solution.append(int(t[k].varValue))
    #print("solution: ", solution)
    #print("Value: ", value(model.objective))
    
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
        
    print(V)
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
    best_bound, best_solution, status = logs_dict["best_bound"], logs_dict["best_solution"], logs_dict["status"]

    remove("log_info-3"+str(ind)+".log")
    
    return best_solution, ceil(best_bound), int(status == "MIP - Integer optimal") 

########## end of ILP-3 model implementation 
    

if __name__ == '__main__':

    # terminal parameters in the program
    # Calling procedure python3 cplex_csp_ilp.py -ifile <string_to_the-file> -time <double> -ilp <number> -ofile <string-path-to-out-saving-res>   
    
    # DEFAULT params and global vars:
    char_to_int = {}
    timelimit: float = 60.0 #sec
    model_num: int = 3 # ILP-1 or ILP-3 for now implemented
    input_file = "4_10_600.virus"
    out_file = "input_file" + ".out" 
    
    ### read terminal arguments:
    
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"ht:i:o:m:",["time=", "ifile=","ofile=", "model="])
    for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile> -t <timelimit> -m  <ILP-model-applied>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
           input_file = arg
      elif opt in ("-o", "--ofile"):
           out_file = arg
      elif opt in ("-t", "--time"):
           timelimit = arg
      elif opt in ("-m", "--model"):
           model_num = arg
           
    print ('Input file is ', input_file)
    print ('Output file is ', out_file)
    print ('Time: ', timelimit)
    print("model num: ", model_num)
    
    ### end reading terminal params
    
    strings, alphabet = read_instance(input_file)
    # preprocessing:
    strings_converted = conversion_chars_to_nums(strings) 
    
    ### Run ILP solver (CPLEX):
    curr = time()
    if model_num == 1:
    	primal, dual, status = ILP1(strings_converted, alphabet, timelimit, 0)
    else:
    	primal, dual, status = ILP3(strings_converted, alphabet, timelimit, 0)
    	
    time_execute = time() - curr
    print("Solution: ", primal, "\nDual bound: ",  dual, "\nStatus: ", status, "\nTime: ", time_execute)
    # Save into a file: 
    f = open(out_file, "w")
    string_stat = "Solution: " + str(primal) + " Dual bound: " +  str(dual) + " Status: " + str(status) +" Time: " + str(time_execute)
    print(string_stat)
    f.write(str(string_stat))
    f.close()
    
