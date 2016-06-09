
def readLines (pathName):
    with open(pathName) as lines:
        return map(lambda line: line.strip(), lines)

