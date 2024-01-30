import random
import os

def oneSeqGenerator(alphabet, length):
    seq = ""
    for _ in range(length):
        seq += random.choice(alphabet)
    return seq

def seqGenerator(alphabet, nstrings, length):
    seqs = []
    for _ in range(nstrings):
        seq = oneSeqGenerator(alphabet, length)
        seqs.append(seq)
    return seqs

def writeToFile(seqs, alphabet, nstring, length):
    fname = str(len(alphabet)) + "_" + str(nstring) + "_" + str(length)
    file = open(fname, 'w')
    for seq in seqs:
        file.write(seq)
        file.write('\n')
    file.close()
    return

def generateDataSet(alphabets, nstrings, lengths):
    for alphabet in alphabets:
        for nstring in nstrings:
            for length in lengths:
                seqs = seqGenerator(alphabet, nstring, length)
                writeToFile(seqs, alphabet, nstring, length)
    return

if __name__ == "__main__":
    os.chdir("C:/Users/Alireza/Documents/Research/Closest String Problem/Implementation/test cases")
    alphabets = [
    ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
    ]
    nstrings = [10, 15, 20, 25, 40, 60, 80, 100]
    lengths = [50, 100, 150, 200, 250, 300, 350, 400]
    generateDataSet(alphabets, nstrings, lengths)