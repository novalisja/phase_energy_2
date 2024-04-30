# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:46:07 2022

@author: novalis
"""

import segyio
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from skimage import filters
from skimage import color, morphology
import cv2
from phasepack import phasecong,phasesym

def seis_cube(file,nx,ni,ns):
    size_data=[nx*1,ni*1,ns*1]
    seis=np.fromfile(file,dtype=np.float32)
    seis=np.reshape(seis,size_data)
    return seis
file1="c:/Users/novalis/work/datasa/issap20_AI/issap20_Pp.sgy"
file2="c:/Users/novalis/work/datasa/issap20_AI/issap20_Fault.sgy"
file3='c:/Users/novalis/work/datasa/geoframe_demo_raw.dat'
file4='c:/Users/novalis/work/datasa/geoframe_demo_detectII.dat'
file4s='c:/Users/novalis/work/datasa/geoframe_demo_raw_Skeleton.dat'

file5='c:/Users/novalis/work/datasa/mig_gtd_spq.dat'

with segyio.open(file1,xline=181) as filesgy:
    seismic=segyio.tools.cube(filesgy)

with segyio.open(file2,xline=181) as filesgy:
    mask=segyio.tools.cube(filesgy)

# Create two subplots and unpack the output array immediately
def imgs_show(im1,im2,title1='image 1',title2='image 2'):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(im1,cmap='gray')
    ax1.set_title(title1)
    ax2.imshow(im2,cmap='gray')
    ax2.set_title(title2)

def im_rescale(img,mi=0,ma=1):
    imi=img.min()
    ima=img.max()
    d=ima-imi
    imgr=mi+(img-imi)*ma/d
    return imgr
    
def img_show(im,cmp='gray'):
    plt.imshow(im,cmap=cmp)
    


def seis_show(im):
    plt.imshow(im,cmap=plt.cm.seismic)


def rem_small_objs1(image,disk=1):
    footprint=morphology.disk(disk)
    res= morphology.white_tophat(image, footprint)
    return image-res

def rem_small_objs2(image,min_size=64,con=1):
    imager=im_rescale(image)*255 
    imager=imager.astype(int)
    res= morphology.remove_small_objects(imager,min_size,connectivity=con)
    return imager-res    

xl,il,ns=seismic.shape

geo=seis_cube(file3,271,221,876)
geo_de=seis_cube(file4,271,221,876)
geo_sk=seis_cube(file4s,271,221,876)

mig=seis_cube(file5,565,620,301)

i=50    
seis_im50=seismic[i,:,:].T
mask_im50=mask[i,:,:].T

np.save('seis_im50',seis_im50)
np.save('mask_im50',mask_im50)

seis_show(seis_im50)
img_show(seis_im50)    
img_show(mask_im50)

imgs_show(im, ma)    

sobel_h=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sobel_v=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

average= np.ones((5,5),np.float32)/25

dst = cv2.filter2D(im,-1,average)
imgs_show(im,dst)  


dsth = cv2.filter2D(im,-1,sobel_h)
imgs_show(im,dsth)  

dstv = cv2.filter2D(im,-1,sobel_v)
imgs_show(im,dstv)  

mag=np.sqrt(dstv*dstv+dsth*dsth)

imgs_show(im,mag)  

imgeo=geo[:,:,273]
plt.imshow(imgeo)

imgeo_de=geo_de[:,:,273]
plt.imshow(imgeo_de)

imgeo_sk=geo_sk[:,:,273]
img_show(imgeo_sk)


#M, m, ori, ft, PC, EO,T


from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb



def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')





METHODS = ['default','ror','uniform','nri_uniform','var']
METHOD = METHODS[2]
radius = 3
n_points = 3 * radius
lbp = local_binary_pattern(im, n_points, radius, METHOD)
imgs_show(im,lbp)

lbp_geo = local_binary_pattern(imgeo, n_points, radius, METHOD)
imgs_show(imgeo,lbp_geo)

lbp_geo_de = local_binary_pattern(imgeo_de, n_points, radius, METHOD)
imgs_show(imgeo_de,lbp_geo_de)

lbp_geo_sk = local_binary_pattern(imgeo_sk, n_points, radius, METHOD)
imgs_show(imgeo_sk,lbp_geo_sk)



im_mig=mig[:,:,181]


in_img=im_mig
in_img=im
in_img=imgeo
in_img=imgeo_de
in_img=imgeo_sk


M, m, ori, ft, PC, EO,T=phasecong(in_img, nscale=5, norient=6, minWaveLength=3,\
        mult=2.1,sigmaOnf=0.55, k=2., cutOff=0.5, g=10., noiseMethod=-1)

phaseSym, orientation, totalEnergy, T=phasesym(in_img, nscale=5, norient=6,minWaveLength=3, \
         mult=2.1,sigmaOnf=0.55, k=2., polarity=0, noiseMethod=-1)

imgs_show(in_img,M)
imgs_show(in_img,phaseSym)
imgs_show(in_img,totalEnergy)
    

    
METHOD = METHODS[1]
lbp_mig = local_binary_pattern(M, n_points, radius, METHOD)
imgs_show(M,lbp_mig)


imgs_show(imgeo,imgeo_de)
imgs_show(imgeo,imgeo_sk)

METHOD = METHODS[1]
lbp_mig = local_binary_pattern(phaseSym, n_points, radius, METHOD)
imgs_show(phaseSym,lbp_mig)

METHOD = METHODS[3]
lbp_mig = local_binary_pattern(phaseSym, n_points, radius, METHOD)
imgs_show(phaseSym,lbp_mig)




w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

plt.imshow(overlay_labels(imgeo_de, lbp_geo_de, edge_labels))
plt.imshow(overlay_labels(imgeo_de, lbp_geo_de, flat_labels))
plt.imshow(overlay_labels(imgeo_de, lbp_geo_de, corner_labels))

plt.imshow(overlay_labels(phaseSym, lbp_mig, edge_labels))
plt.imshow(overlay_labels(phaseSym, lbp_geo_de, flat_labels))
plt.imshow(overlay_labels(phaseSym, lbp_geo_de, corner_labels))


