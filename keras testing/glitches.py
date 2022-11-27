import numpy as np
import scipy.ndimage
import cv2
import os
import time
import skimage.exposure
from numpy.random import default_rng

def glitch1(img):
    return img

def glitch2(sharp_image):
    
    def gaussian_blur(sharp_image, sigma):
        # Filter channels individually to avoid gray scale images
        blurred_image_r = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 0], sigma=sigma)
        blurred_image_g = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 1], sigma=sigma)
        blurred_image_b = scipy.ndimage.filters.gaussian_filter(sharp_image[:, :, 2], sigma=sigma)
        blurred_image = np.dstack((blurred_image_r, blurred_image_g, blurred_image_b))
        return blurred_image


    def uniform_blur(sharp_image, uniform_filter_size):
        # The multidimensional filter is required to avoid gray scale images
        multidim_filter_size = (uniform_filter_size, uniform_filter_size, 1)
        blurred_image = scipy.ndimage.filters.uniform_filter(sharp_image, size=multidim_filter_size)
        return blurred_image


    def blur_image_locally(sharp_image, mask, use_gaussian_blur, gaussian_sigma, uniform_filter_size):

        one_values_f32 = np.full(sharp_image.shape, fill_value=1.0, dtype=np.float32)
        sharp_image_f32 = sharp_image.astype(dtype=np.float32)
        sharp_mask_f32 = mask.astype(dtype=np.float32)

        if use_gaussian_blur:
            blurred_image_f32 = gaussian_blur(sharp_image_f32, sigma=gaussian_sigma)
            blurred_mask_f32 = gaussian_blur(sharp_mask_f32, sigma=gaussian_sigma)

        else:
            blurred_image_f32 = uniform_blur(sharp_image_f32, uniform_filter_size)
            blurred_mask_f32 = uniform_blur(sharp_mask_f32, uniform_filter_size)

        blurred_mask_inverted_f32 = one_values_f32 - blurred_mask_f32
        weighted_sharp_image = np.multiply(sharp_image_f32, blurred_mask_f32)
        weighted_blurred_image = np.multiply(blurred_image_f32, blurred_mask_inverted_f32)
        locally_blurred_image_f32 = weighted_sharp_image + weighted_blurred_image

        locally_blurred_image = locally_blurred_image_f32.astype(dtype=np.uint8)

        return locally_blurred_image
    
    
    height, width, channels = sharp_image.shape
    sharp_mask = np.full((height, width, channels), fill_value=1)
    
    t=int(time.ctime().split(' ')[-2].split(':')[-1])
    
    seed=np.random.randint(1,10)
    xoffset=np.random.randint(2,6)
    yoffset=np.random.randint(2,6)
    if(t%2==0):
        sharp_mask[int(height / seed): int(xoffset * height / seed), int(width / seed): int(yoffset * width / seed), :] = 0
    else:
        sharp_mask[int(height / seed): int(yoffset*height / seed), int(width / seed): int(xoffset*width / seed), :] = 0
    
    result = blur_image_locally(
            sharp_image,
            sharp_mask,
            use_gaussian_blur=True,
            gaussian_sigma=31,
            uniform_filter_size=201)
    return result
    

def glitch3(img):
    height, width = img.shape[:2]
    t=int(time.ctime().split(' ')[-2].split(':')[-1])
    x=np.random.randint(1,35)
    y=np.random.randint(1,35)
    if(y%2==0):
        seedval = t
    else:
        seedval = x
    rng = default_rng(seed=seedval)
    noise = rng.integers(0, 255, (height,width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])
    result1 = cv2.add(img, mask)
    edges = cv2.Canny(mask,50,255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges = cv2.merge([edges,edges,edges])
    result2 = result1.copy()
    result2[np.where((edges == [255,255,255]).all(axis=2))] = [0,0,0]
    noise = cv2.merge([noise,noise,noise])
    result3 = result2.copy()
    result3 = np.where(mask==(255,255,255), noise, result3)
    cv2.imwrite('lena_random_blobs1.jpg', result1)
    cv2.imwrite('lena_random_blobs2.jpg', result2)
    cv2.imwrite('lena_random_blobs3.jpg', result3)
    return result3

def glitch4(img):
    height, width = img.shape[:2]
    t=int(time.ctime().split(' ')[-2].split(':')[-1])
    x=np.random.randint(1,35)
    y=np.random.randint(1,35)
    if(y%2==0):
        seedval = t
    else:
        seedval = x
    rng = default_rng(seed=seedval)
    noise = rng.integers(0, 255, (height,width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])
    result1 = cv2.add(img, mask)
    edges = cv2.Canny(mask,50,255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges = cv2.merge([edges,edges,edges])
    result2 = result1.copy()
    result2[np.where((edges == [255,255,255]).all(axis=2))] = [0,0,0]
    noise = cv2.merge([noise,noise,noise])
    result3 = result2.copy()
    
    #result3 = np.where(mask==(255,255,255), noise, result3)
    cv2.imwrite('lena_random_blobs1.jpg', result1)
    cv2.imwrite('lena_random_blobs2.jpg', result2)
    cv2.imwrite('lena_random_blobs3.jpg', result3)
    #cv2.imshow('resu;t',result3)
    #cv2.imshow('resu;t',result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result3

def glitch5(img):
    width = 5000
    x=int((width-1920)/2)
    height = img.shape[0]
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resizedt=resized[:,x:1920+x,:]
    return resizedt


        
        

