
def parseFile(filename):
    output = []
    for row in open(filename):
        rowSplitted = []
        for element in row.split():
            rowSplitted.append(float(element))
        output.append(rowSplitted)
    return output