
def scale(x):
    maxInput = x[0].copy()
    minInput = x[0].copy()
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            if( maxInput[j] < x[i][j] ):
                maxInput[j] = x[i][j]
            if( minInput[j] > x[i][j] ):
                minInput[j] = x[i][j]
    elems = []
    for elem in x:
        elemToAdd = []
        for i in range(0, len(x[0])):
            if maxInput[i] != minInput[i]:
                elemToAdd.append((elem[i]-minInput[i])/(maxInput[i]-minInput[i]))
        elems.append(elemToAdd)
    return elems.copy()