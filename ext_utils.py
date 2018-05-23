
import numpy as np
import math

def makePyramidRegions(im, bins):
    '''
    the input should be (H x W x K) ndarray
    '''
    # print im.shape # c,h,w
    h = im.shape[0]
    w = im.shape[1]
    
    numLevels = bins.shape[0]
    #numRegions = np.sum(np.prod(bins, axis=1))
    regions = []
    for level in range(numLevels):
        regionH = int(math.floor(h / bins[level, 0]))
        regionW = int(math.floor(w / bins[level, 1]))

        for vIdx in range(0, h - regionH + 1, regionH):
            for hIdx in range(0, w - regionW + 1, regionW):
                region = im[vIdx:vIdx + regionH, hIdx:hIdx + regionW, :]
                regions.append(region)
    return regions
