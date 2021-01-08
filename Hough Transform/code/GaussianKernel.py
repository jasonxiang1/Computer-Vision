import numpy as np

def Gauss2D(kernel=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in kernel]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

if __name__ == "__main__":
    # execute only if run as a script
    print(Gauss2D())