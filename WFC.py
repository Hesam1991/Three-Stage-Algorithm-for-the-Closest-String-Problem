import pickle
import random
import os

def calculateLetterFreq(inputStrings, alphabet):
    """
    :param inputStrings:  list of input strings, all strings ar of  the same length
    :param alphabet: alphabet used to create the input strings
    """

    # find string length from the first string
    stringLength = len(inputStrings[0])

    # letter frequency table is a 2-D array implemented as list of list
    letterFreqTable = {}
    for position in range(stringLength):
        # create frequency for all alphabet letter at this position
        alphabetFreqTable = {}

        for alphabetLetter in alphabet:
            alphabetFreqTable[alphabetLetter] = 0
            for inputString in inputStrings:
                if inputString[position] == alphabetLetter:
                    alphabetFreqTable[alphabetLetter] += 1

        letterFreqTable[position] = alphabetFreqTable

    return letterFreqTable

def calculateScoreboard(letterFreqTable, positions):
    scoreboard = []
    for position in positions:
        alphabetFreqTable = letterFreqTable[position]
        for letter, freq in alphabetFreqTable.items():
            score = (position, letter, freq)
            scoreboard.append(score)

    scoreboard.sort(key=lambda entry: entry[2], reverse=True)
    return scoreboard

def calculateDistance(string1, string2):
    """
    find the Hamming distance between string1 and string 2
    :param string1: string 1
    :param string2: string 2
    :return:
    """

    distance = 0
    stringLength = len(string1)
    for index in range(stringLength):
        if string1[index] != string2[index]:
            distance += 1

    return distance

def calculateDistancesWithInputStrings(answer, inputStrings):
    distances = []
    for inputStringIndex in range(len(inputStrings)):
        distance = calculateDistance(answer, inputStrings[inputStringIndex])
        distances.append((inputStringIndex, distance))

    distances.sort(key=lambda entry: entry[1], reverse=True)
    return distances

def getMaxDistStr(strDistArray):
    maxDist=strDistArray[0][1]
    count = 0
    # print("strDistArray", strDistArray)
    for string, distance in strDistArray:
        if distance == maxDist:
            count += 1
    pick = random.randrange(count)
    return strDistArray[pick]

def findMaxLetters(maxDistanceInputString, scoreboard):
    maxLetters = []
    for position, letter, score in scoreboard:
        if maxDistanceInputString[position] == letter:
          maxLetters.append([position, letter, score])
    maxDist = maxLetters[0][2]
    maxDistLetters = []
    for letter in maxLetters:
        if letter[2] == maxDist:
            maxDistLetters.append(letter)
    return random.choice(maxDistLetters)

def findClosestString(alphabet, inputStrings, maximumDistance):
    # all string are of same length
    stringLength = len(inputStrings[0])

    letterFreqTable = calculateLetterFreq(inputStrings, alphabet)

    # create initial answer with all SPACE
    answer = [" "] * stringLength
    undecidedPositions = set(range(stringLength))

    while len(undecidedPositions) > 0:
        # there is at least 1 undecided positions
        # print()
        # print("current undecided position:", len(undecidedPositions), undecidedPositions)

        scoreboard = calculateScoreboard(letterFreqTable, undecidedPositions)
        # print("current scoreboard:", scoreboard)

        # sort and find string with maximum distance to current answer
        inputStringDistances = calculateDistancesWithInputStrings(answer, inputStrings)

        # first string is the one with maximum distance to the answer
        maxDistanceInputStringIndex, maxDistance = getMaxDistStr(inputStringDistances)
        maxDistanceInputString = inputStrings[maxDistanceInputStringIndex]
        # print("current string with maximum Distance: ", maxDistanceInputString, maxDistance)

        # find the 1st (position, letter) in scoreboard  that matches
        # maxDistanceInputString

        maxLetter = findMaxLetters(maxDistanceInputString, scoreboard)
        # print("maxLetter", maxLetter)# found, update the answer
        # print("found position %d, letter '%s'" % (maxLetter[0], maxLetter[1]))
        #maxLetter has [position, letter, score]
        answer[maxLetter[0]] = maxLetter[1]
        # print("answer = ", answer)

        # remove position from undecided positions
        undecidedPositions.remove(maxLetter[0])

    return answer

def fileRead(Fname, lenstrings):
    F = open(Fname, 'r')
    S = []
    for _ in range(lenstrings):
        S.append(F.readline().strip())
    return S




_path = "C:/Users/Alireza/Documents/Research/Closest String Problem/Implementation"
os.chdir(_path)
S = fileRead("Tdata.txt", 10)
# alphabet = ['A', 'C', 'G', 'T']
alphabet = ['0', '1']
# alphabet = ['A', 'C', 'D','E','F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'X', 'Y']
# alphabet = ['a', 'c', 'g', 't']
x = findClosestString(alphabet, S, 60)
fstr = ""
for i in range(len(x)):
    fstr += x[i]
ans = []
for i in range(len(S)):
    ans.append(calculateDistance(fstr, S[i]))

print(max(ans))