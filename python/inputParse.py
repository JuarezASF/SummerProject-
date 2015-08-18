import sys

def parseInput(inputList):
    if len(inputList) > 1:
        inputList =  inputList[1:]
    inputDict = dict()
    for x in inputList:
        x = x.split("=")
        if len(x) == 2:
            first, second = x
            inputDict[first] = second
        else:
            first = x[0]
            inputDict[first] = None
    return inputDict


if __name__ == '__main__':

    print parseInput(sys.argv)
