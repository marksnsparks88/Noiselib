import numpy as np
import aalib
import argparse

########## - value noise
##Basic gaussian filter
def gaussian(arr, depth, noise=True):
    if noise == True:
        input_ = np.random.normal(size=(arr.shape[0], arr.shape[1]))
    else:
        input_ = arr.copy()
    for i in range(1, depth+1):
        arr += ndimage.filters.gaussian_filter(input_, int(i)) * i**2
    return arr

######### - gradient noise
#basic gradient noise
def gradient(width, height, cells, pos=np.array((0, 0)),
             point_order=[0, 1, 2, 3], rndx=[127.1, 127.1, 127.1, 127.1], 
             rndy=[311.7, 311.7, 311.7, 311.7], rnd_off=43758.5453123):
    v = np.indices((width, height))
    vy = (v[1]/v.shape[2] + pos[1]) * cells
    vx = (v[0]/v.shape[1] + pos[0]) * cells
    ix = np.floor(vx)
    iy = np.floor(vy)
    fx = aalib.decimal(vx)
    fy = aalib.decimal(vy)
    
    p = [ '' for _ in range(4)]
    p[point_order[0]] = aalib.rand1d(ix, iy, rnd_off, rndx=rndx[0], rndy=rndy[0])
    p[point_order[1]] = aalib.rand1d(ix + 1, iy + 0, rnd_off, rndx=rndx[1], rndy=rndy[1])
    p[point_order[2]] = aalib.rand1d(ix + 0, iy + 1, rnd_off, rndx=rndx[2], rndy=rndy[2])
    p[point_order[3]] = aalib.rand1d(ix + 1, iy + 1, rnd_off, rndx=rndx[3], rndy=rndy[3])
    sx = aalib.smoothstep(fx)
    sy = aalib.smoothstep(fy)
    return aalib.lerp(p[0], p[1], sx) + (p[2] - p[0]) * sy * (1-sx) + (p[3] - p[1]) * sx * sy
    
#perlin noise
def perlin(width, height, cells, curve="QIC", pos=np.array((0.0, 0.0))):
    v = np.indices((width, height))
    vx = (v[1]/v.shape[2] + pos[1]) * cells
    vy = (v[0]/v.shape[1] + pos[0]) * cells
    ix = np.floor(vy)
    iy = np.floor(vx)
    fx = aalib.decimal(vy)
    fy = aalib.decimal(vx)
    
    ux = aalib.smoothstep(fx, version=curve)
    uy = aalib.smoothstep(fy, version=curve)
    
    av = np.transpose(aalib.rand2d(ix+np.zeros_like(ix), iy+np.zeros_like(iy)), (2, 0, 1))
    af = np.array((fx - np.zeros_like(fx), fy - np.zeros_like(fy)))
    ad = aalib.dot2d(av[0], av[1], af[0], af[1])
    
    bv = np.transpose(aalib.rand2d(ix+np.ones_like(ix), iy+np.zeros_like(iy)), (2, 0, 1))
    bf = np.array((fx - np.ones_like(fx), fy - np.zeros_like(fy)))
    bd = aalib.dot2d(bv[0], bv[1], bf[0], bf[1])
    
    cv = np.transpose(aalib.rand2d(ix+np.zeros_like(ix), iy+np.ones_like(iy)), (2, 0, 1))
    cf = np.array((fx - np.zeros_like(fx), fy - np.ones_like(fy)))
    cd = aalib.dot2d(cv[0], cv[1], cf[0], cf[1])
    
    dv = np.transpose(aalib.rand2d(ix+np.ones_like(ix), iy+np.ones_like(iy)), (2, 0, 1))
    df = np.array((fx - np.ones_like(fx), fy - np.ones_like(fy)))
    dd = aalib.dot2d(dv[0], dv[1], df[0], df[1])
    
    return aalib.lerp( aalib.lerp(ad, bd, ux), aalib.lerp(cd, dd, ux), uy)


#simplex_noise
#permute provides the randomness
def permute(x, modulo):
    r = (((x*34.0)+1.0)*x)%modulo
    return r

def simplex(width, height, cells, off=np.array((0.0, 0.0))):
    v = np.indices((width, height))
    vx = (v[1]/v.shape[2] + off[1]+1e-10) * cells
    vy = (v[0]/v.shape[1] + off[0]) * cells
    
    C = np.array(( 0.211324865405187,       ## (3.0-np.sqrt(3.0))/6
                   0.366025403784439,       ## 0.5*(np.sqrt(3.0)-1)
                   -0.577350269189626,      ## -1.0 + 2.0 * C[0]
                   0.024390243902439025 ))  ## 1/41
        
    #first corner p0
    idtC1 = aalib.dot2d(vx, vy, C[1], C[1])
    ix = np.floor(vx+idtC1)
    iy = np.floor(vy+idtC1)
    
    idtC0 = aalib.dot2d(ix, iy, C[0], C[0])
    p0x = vx - ix + idtC0
    p0y = vy - iy + idtC0
    
    #other corners p1, p2
    i1x = (p0x > p0y)+0.0 ## if x > y then 1, 0
    i1y = (p0y > p0x)+0.0 ## if y > x then 0, 1
        
    p1x = p0x + C[0] - i1x
    p1y = p0y + C[0] - i1y
    
    p2x = p0x + C[2]
    p2y = p0y + C[2]
    
    modulo = 289
    ix = ix%modulo
    iy = iy%modulo
    
    permx = permute(permute(iy + 0.0, modulo)+ ix + 0.0, modulo)
    permy = permute(permute(iy + i1y, modulo)+ ix + i1x, modulo)
    permz = permute(permute(iy + 1.0, modulo)+ ix + 1.0, modulo)
    
    ## max(0.5 - (dot(p0, p0), dot(p1, p1), dot(p2, p2)), 0.0)
    dtp0max = 0.5 - aalib.dot2d(p0x, p0y, p0x, p0y)
    dtp1max = 0.5 - aalib.dot2d(p1x, p1y, p1x, p1y)
    dtp2max = 0.5 - aalib.dot2d(p2x, p2y, p2x, p2y)
    
    dtp0max[dtp0max < 0.0] = 0.0
    dtp1max[dtp1max < 0.0] = 0.0
    dtp2max[dtp2max < 0.0] = 0.0
    
    mx = dtp0max**4
    my = dtp1max**4
    mz = dtp2max**4
    
    permx = permx * 1
    permy = permy * 1
    permz = permz * 1
    
    xx = 2.0 * aalib.decimal(permx*C[3]) - 1.0
    hx = abs(xx) - 0.5
    oxx = np.floor(xx+0.5)
    a0x = xx - oxx
    
    xy = 2.0 * aalib.decimal(permy*C[3]) - 1.0
    hy = abs(xy) - 0.5
    oxy = np.floor(xy+0.5)
    a0y = xy - oxy
    
    xz = 2.0 * aalib.decimal(permz*C[3]) - 1.0
    hz = abs(xz) - 0.5
    oxz = np.floor(xz+0.5)
    a0z = xz - oxz
    
    mx = mx * (1.79284291400159 - 0.85373472095314 * (a0x**2+hx**2))
    my = my * (1.79284291400159 - 0.85373472095314 * (a0y**2+hy**2))
    mz = mz * (1.79284291400159 - 0.85373472095314 * (a0z**2+hz**2))
    
    gx = a0x * p0x + hx * p0y
    gy = a0y * p1x + hy * p1y
    gz = a0z * p2x + hz * p2y
    
    p_ = aalib.dot3d(mx, my, mz, gx, gy, gz)
    
    return 130 * p_

# fractal brownian motion
def fbm(width, height, amp=0.5, gain=0.5, octaves=6,
           range_=2, cell_off=0, rotation=0,
           off=np.array((0.0, 0.0))):
    
    n = np.zeros((width, width))
    cells_ = 1 + cell_off
    amplitude = amp
    offset = off
    for o in range(octaves):
        n = n + amplitude * simplex(width, height, cells_, off=offset)
        cells_ *= range_
        amplitude *= gain
        offset = aalib.rotate2d(rotation, offset)
    return n

############ cellular noise
# basic cellular
def cellular(width, height, points):
    v = np.indices((width, height))
    vx = v[1]
    vy = v[0]
    lp = np.ndarray((width, height))
    lp.fill(1e10)
    for _ in range(points):
        rx = np.random.randint(0, width)
        ry = np.random.randint(0, height)
        p = np.sqrt((vx - rx)**2 + (vy - ry)**2)
        
        pt  = (p <= lp)+0
        pt2 = (lp < p)+0
        
        lp = (p * pt) + (lp * pt2)
    
    return lp

# Voronoi noise
def voronoi(ncells, res):
    ndx = np.indices((res, res))
    ndxx = ndx[1]
    ndxy = ndx[0]
    
    cells = [[ np.full((res, res), 1e+5) for _ in range(ncells)] for _ in range(ncells)]
    px = [[np.random.randint(0, res) for _ in range(ncells)] for _ in range(ncells)]
    py = [[np.random.randint(0, res) for _ in range(ncells)] for _ in range(ncells)]
    
    for i in range(ncells):
        for j in range(ncells):
            for y_ in range(3):
                for x_ in range(3):
                    y = (i + (y_ - 1)) % ncells
                    x = (j + (x_ - 1)) % ncells
                    
                    px_ = (res*(x_ - 1)) + px[y][x]
                    py_ = (res*(y_ - 1)) + py[y][x]
                    
                    a = (ndxx - px_ )**2
                    b = (ndxy - py_ )**2
                    
                    d = np.sqrt(a+b)
                    #d = a+b
                    
                    pt  = (d <= cells[i][j])+0
                    pt2 = (cells[i][j] < d) +0
                    
                    cells[i][j] = (d * pt) + (cells[i][j] * pt2)
                    
    
    arry = []
    for i in range(len(cells)):
        arrx = np.hstack((cells[i]))
        try:
            arry = np.vstack((arry, arrx))
        except:
            arry = arrx.copy()
        
    return arry


if __name__ == "__main__":
    pass
