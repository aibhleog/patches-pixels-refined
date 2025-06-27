'''

Using the code from patches-pixels.py, this code works to account for the pixels
on the edge of the regions where there is only partial coverage.


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse,Circle,Wedge,Rectangle
from astropy.convolution import Gaussian2DKernel
import sys


# making a fake galaxy
galaxy = Gaussian2DKernel(4).array 
xcen,ycen = 18,20

scale = 5 # this will convert each pixel into a (scale X scale) grid



# just wrote this into a function to make life easier
def get_points(mypatch,size,radius=0):
    '''
    INPUTS:
    >> mypatch ---- the matplotlib patch shape
    >> size ------- integer, the size in pixels of one 
                    side of the "galaxy" image
    >> radius ----- the radius wiggle room
                    
    OUTPUTS:
    >> points ----- the list of valid x,y coordinates
                    that overlap with the patch
    '''
    # create a list of possible coordinates
    x,y = np.arange(0,size),np.arange(0,size)

    g = np.meshgrid(x,y)
    coords = list(zip(*(c.flat for c in g)))

    # create the list of valid coordinates (from patch)
    points = np.vstack([p for p in coords if mypatch.contains_point(p, radius=radius)])
    return np.array(points)


def region_overlap(points1,points2):
    '''
    assumes you want to know how many points2 are in points1, with the final array being a binary sort of points2
    https://stackoverflow.com/questions/55434338/check-if-array-is-part-of-a-bigger-array
    '''
    isit = np.zeros(len(points2)).astype(int)
    for i,coord in enumerate(points2):
        if (points1 == coord).all(axis=1).any():
            isit[i] = 1
    return isit
    
    

def rebin(a,newshape):
    '''
    Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)
    
    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]


def mapping_pixel(x,y,scale=1):
    '''
    Takes a pixel coordinate from the original map and creates a 
    Rectangle patch that has nXn points (based upon the scale #).
    
    '''
    new_x,new_y = x*scale-0.5,y*scale-0.5
    return Rectangle((new_x,new_y), 1*scale, 1*scale, alpha=0.3,facecolor='r')




# ---------------------------------
# plotting different patch examples
# ---------------------------------

f = plt.figure(figsize=(13,8))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1])
ax2 = plt.subplot(gs[0])

ax2.imshow(galaxy,origin='lower',cmap='Blues') # the fake galaxy

# making a circular aperture
circle = Circle((xcen,ycen),
                radius = 5,
                alpha = 0.3,
                facecolor = 'C1')

points = get_points(circle,len(galaxy),radius=0.5)

ax2.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)
ax2.add_patch(circle) # make sure patch is added to plot last

for x,y in [[20,20],[22,23],[23,21]]:
    rectangle = mapping_pixel(x,y)
    ax2.add_patch(rectangle)

ax2.set_ylim(17.5,24)
ax2.set_xlim(17.5,24)
lims = ax2.get_xlim()



# rebinning to more points for the refined part!
rebin_galaxy = rebin(galaxy.copy(), np.asarray(galaxy.shape)*scale)





# looking at finer sampling
ax2 = plt.subplot(gs[1])
ax2.imshow(rebin_galaxy,origin='lower',cmap='Blues') # rebinned fake galaxy

# the same aperture but scaled to the new grid
circle = Circle(((xcen+0.5)*scale-(1/scale),(ycen+0.5)*scale-(1/scale)),
                radius = 5*scale,
                alpha = 0.3,
                facecolor = 'C1')

points = get_points(circle,len(rebin_galaxy),radius=0.5/scale)
ax2.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)

# example pixels from before
for x,y in [[20,20],[22,23],[23,21]]:
    rectangle = mapping_pixel(x,y,scale)
    # counting points covered
    pix_points = get_points(rectangle,len(rebin_galaxy),radius=0.5/scale)
    overlap_index = region_overlap(points,pix_points)
    print(f'out of {scale**2} points in one pixel, there are {len(pix_points[overlap_index==1])} covered by the region.')
    
    ax2.add_patch(rectangle)
    

ax2.add_patch(circle) # make sure patch is added to plot last
ax2.set_ylim((lims[0]+0.5)*scale-(1/scale),(lims[1]+0.5)*scale-(1/scale))
ax2.set_xlim((lims[0]+0.5)*scale-(1/scale),(lims[1]+0.5)*scale-(1/scale))

plt.tight_layout()
plt.show()
plt.close('all')




# trying it out for all pixels, then making map scaling by percent coverage
print('\nRunning on all pixels covered by region...',end='\n\n')

# making a slightly wider radius so I can have all the edge pixels included
circle = Circle((xcen,ycen),radius = 5,alpha = 0.3,facecolor = 'C1')
check_points = get_points(circle,len(galaxy),radius=1)

# same points but on the larger scaled grid
circle2 = Circle(((xcen+0.5)*scale-(1/scale),(ycen+0.5)*scale-(1/scale)),
                radius = 5*scale,alpha = 0.3,facecolor = 'C1')
check_rebin_points = get_points(circle2,len(rebin_galaxy),radius=1)

# empty array
coverage = np.zeros_like(galaxy)

# running through all of the coordinates in the ORIGINAL grid
for i,coord in enumerate(check_points):
    if i % 15 == 0: print(f'At coordinate {i}/{len(check_points)}')
    
    x,y = coord # in original pixel coords
    rectangle = mapping_pixel(x,y,scale) # in scaled pixel coords, 1 pixel
    
    # counting points covered
    pix_points = get_points(rectangle,len(rebin_galaxy),radius=0.5/scale)
    overlap_index = region_overlap(check_rebin_points,pix_points)
    total_covered = len(pix_points[overlap_index==1])
    coverage[y,x] = total_covered / scale**2
    

# we don't care about the non-aperture pixels
coverage[coverage==0] = np.nan


# looking at coverage
f = plt.figure(figsize=(9.5,4.55))
gs = gridspec.GridSpec(1,2,width_ratios=[1,1.177],wspace=0)

ax2 = plt.subplot(gs[0])
ax2.imshow(galaxy,origin='lower',cmap='Blues')

circle = Circle((xcen,ycen),radius = 5,alpha = 0.3,facecolor = 'C1')
ax2.scatter(check_points[:,0],check_points[:,1],color='k',s=10,alpha=0.8,zorder=10)
ax2.add_patch(circle) # make sure patch is added to plot last

ax2.set_xticklabels([])
ax2.set_yticklabels([])


# COVERAGE
ax2 = plt.subplot(gs[1])
ax2.imshow(galaxy,origin='lower',cmap='Blues')

im = ax2.imshow(coverage,origin='lower',cmap='Greens',alpha=0.5)
cbar = plt.colorbar(im,pad=0)
cbar.set_label('coverage fraction',rotation=270,labelpad=20)

ax2.set_xticklabels([])
ax2.set_yticklabels([])


plt.tight_layout()
plt.show()
plt.close('all')



# congrats!  save the coverage array to have your fancy map








