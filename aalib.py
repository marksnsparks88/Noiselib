import numpy as np
from matplotlib import pyplot as plt
# gets decimal numbers from float
def decimal(f):
    return f - np.floor(f)

#cartesian to polar co-ordinates
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    a = np.degrees(np.arctan(y/x))
    return np.array((r, a))

def polar2cart(r, a):
    x = r * np.cos(np.radians(a))
    y = r * np.sin(np.radians(a))
    return np.array((y, x))

#dot product cos numpy version makes no sense
#https://www.cuemath.com/algebra/dot-product/
def dot2d(ax, ay, bx, by):
    return (ax*bx)+(ay*by)

def dot3d(ax, ay, az, bx, by, bz):
    return (ax*bx)+(ay*by)+(az*bz)

#psuedo random
#returns single (scalar) number
def rand1d(x, y, rndx=127.1, rndy=311.7, off=43758.5453123):
    rx = np.zeros_like(x)
    ry = np.zeros_like(y)
    rx.fill(rndx)
    ry.fill(rndy)
    i = dot2d(x, y, rx, ry)
    return -1.0 + 2.0 * decimal(np.sin(i)*off)

#returns 2d vector
def rand2d(x, y, rndx1=127.1, rndy1=311.7, rndx2=269.5, rndy2=183.3, off=43758.5453123):
    rx = np.zeros_like(x)
    ry = np.zeros_like(y)
    rx.fill(rndx1)
    ry.fill(rndy1)
    a = dot2d(x, y, rx, ry)
    rx.fill(rndx2)
    ry.fill(rndy2)
    b = dot2d(x, y, rx, ry)
    
    st = np.array((a, b))
    st = np.transpose(st, (1, 2, 0))
    return -1.0 + 2.0 * decimal(np.sin(st)*off)

#https://en.wikipedia.org/wiki/Smoothstep
def smoothstep(t, version="QIC"):
    if version == "CHC":
        return t * t * (3.0-2.0*t)
    elif version == "QIC":
        return t*t*t*(t*(t*6.-15.)+10.)

#https://en.wikipedia.org/wiki/Linear_interpolation#Programming_language_support
def lerp(a, b, t):
    return a * (1 - t) + b * t

def clerpRGB(arr, *args):
    c = args
    if len(c) > 2:
        R = np.zeros_like(arr)
        G = np.zeros_like(arr)
        B = np.zeros_like(arr)
        for i in range(len(c)-1):
            ts = np.max(arr)/(len(c)-1)
            clp = np.clip(arr, ts * i, ts * (i+1))
            clpt = (arr == clp)+0
            clp = normalize2d(clp, 0, 1)
            R = R + (lerp(c[i][0], c[i+1][0], clp)*clpt)
            G = G + (lerp(c[i][1], c[i+1][1], clp)*clpt)
            B = B + (lerp(c[i][2], c[i+1][2], clp)*clpt)
            
        return np.dstack((R, G, B)).astype(np.uint8)
    else:
        R = arr.copy()
        G = arr.copy()
        B = arr.copy()
        R = lerp(c[0][0], c[1][0], R)
        G = lerp(c[0][1], c[1][1], G)
        B = lerp(c[0][2], c[1][2], B)
    
        return np.dstack((R, G, B)).astype(np.uint8)

#rotation matrix
def rotate2d(degrees, vect):
    r = np.radians(degrees)
    mat = np.array((( np.cos(r), np.sin(-r)),
                    ( np.sin(r), np.cos(r))))
    
    return np.dot(mat, vect)

#matrix normalization
def normalize2d(arr, min_, max_):
    arr = arr - np.min(arr)
    arr = arr / np.max(arr)
    arr = arr * max_ - min_
    arr = arr + min_
    return arr

