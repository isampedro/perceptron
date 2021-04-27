def linearW(eta, g, gder, h, expectedExit, entry, beta):
    return eta*(expectedExit - g(h, beta))*entry

def nonLinearW(eta, g, gder, h, expectedExit, entry, beta):
    return eta*(expectedExit - g(h, beta))*gder(h, beta)*entry