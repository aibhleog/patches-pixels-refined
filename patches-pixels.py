'''

THIS IS THE ORIGINAL SCRIPT COPIED FROM 
https://github.com/aibhleog/patches-pixels/

--------------
This is a quick & dirty way to pick most of the pixels out for given matplotlib patch that you overlay on a galaxy 2D image.


NOTES:  The ring example doesn't work properly if you do a complete  
        ring (ex: theta2=360).  Seems like a bug in the source code.


Reference/inspiration:
https://stackoverflow.com/questions/25145931/extract-coordinates-enclosed-by-a-matplotlib-patch

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse,Circle,Wedge,Rectangle
from astropy.convolution import Gaussian2DKernel


# making a fake galaxy
galaxy = Gaussian2DKernel(4).array 



# just wrote this into a function to make life easier
def get_points(mypatch,size):
    '''
    INPUTS:
    >> mypatch ---- the matplotlib patch shape
    >> size ------- integer, the size in pixels of one 
                    side of the "galaxy" image
                    
    OUTPUTS:
    >> points ----- the list of valid x,y coordinates
                    that overlap with the patch
    '''
    # create a list of possible coordinates
    x,y = np.arange(0,size),np.arange(0,size)

    g = np.meshgrid(x,y)
    coords = list(zip(*(c.flat for c in g)))

    # create the list of valid coordinates (from patch)
    points = np.vstack([p for p in coords if mypatch.contains_point(p, radius=0)])
    return np.array(points)



# ---------------------------------
# plotting different patch examples
# ---------------------------------

f = plt.figure(figsize=(8,8))
gs0 = gridspec.GridSpec(2,2,height_ratios=[1,1],width_ratios=[1,1],
                        wspace=0,hspace=0) 

ax1 = plt.subplot(gs0[0])
ax2 = plt.subplot(gs0[1])
ax3 = plt.subplot(gs0[2])
ax4 = plt.subplot(gs0[3])


# AN ELLIPSE
# ----------
ax1.imshow(galaxy,origin='lower',cmap='Blues')
ax1.text(0.05,0.9,'ELLIPSE',transform=ax1.transAxes,fontsize=15,color='C2')

ellipse = Ellipse((16,16),
                  width = 7,
                  height = 20,
                  angle = 34,
                  alpha = 0.3,
                  facecolor = 'C2')


points = get_points(ellipse,len(galaxy))

ax1.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)
ax1.add_patch(ellipse) # make sure patch is added to plot last


# A CIRCLE
# --------
ax2.imshow(galaxy,origin='lower',cmap='Blues')
ax2.text(0.05,0.9,'CIRCLE',transform=ax2.transAxes,fontsize=15,color='C1')

circle = Circle((18,20),
                radius = 5,
                alpha = 0.3,
                facecolor = 'C1')

points = get_points(circle,len(galaxy))

ax2.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)
ax2.add_patch(circle) # make sure patch is added to plot last


# A RING & WEDGE
# --------------
ax3.imshow(galaxy,origin='lower',cmap='Blues')
ax3.text(0.05,0.9,'RING & WEDGE',transform=ax3.transAxes,fontsize=15,color='C3')

# wedge 1
wedge = Wedge((16,16),
                r = 8,
                theta1 = 0,
                theta2 = 270,
                width = 3,
                alpha = 0.3,
                facecolor = 'C3')

points = get_points(wedge,len(galaxy))

ax3.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)

# wedge 2
wedge2 = Wedge((16,16),
                r = 10,
                theta1 = 300,
                theta2 = 345,
                alpha = 0.3,
                facecolor = 'C3')

points2 = get_points(wedge2,len(galaxy))

ax3.scatter(points2[:,0],points2[:,1],color='k',s=10,alpha=0.8,zorder=10)


ax3.add_patch(wedge) # make sure patch is added to plot last
ax3.add_patch(wedge2) # make sure patch is added to plot last



# A RECTANGLE
# -------
ax4.imshow(galaxy,origin='lower',cmap='Blues')
ax4.text(0.05,0.9,'RECTANGLE',transform=ax4.transAxes,fontsize=15,color='C4')

rectangle = Rectangle((8,13),
                      width = 16,
                      height = 5,
                      alpha = 0.3,
                      facecolor = 'C4')

points = get_points(rectangle,len(galaxy))

ax4.scatter(points[:,0],points[:,1],color='k',s=10,alpha=0.8,zorder=10)
ax4.add_patch(rectangle) # make sure patch is added to plot last



for ax in [ax1,ax2,ax3,ax4]:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.tight_layout()
plt.show()
plt.close('all')