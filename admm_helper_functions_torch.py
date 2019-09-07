#import tensorflow as tf
import numpy as np 
import torch
import torch.nn.functional as F
#

""" Includes helper functions that are used in admm.py and model.py
Last updated: 2/22/2019 

Overview:

    * Padding and cropping functions
    * FFT shifting functions
    * Forward Model (H, Hadj)
    * Soft thresholding functions
    * TV forward/adjoint operators 
"""

###### Complex operations ##########
def complex_multiplication(t1, t2):
    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def complex_abs(t1):
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)

def make_real(c):
    out_r, _ = torch.unbind(c,-1)
    return out_r

def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)

####### Padding and cropping functions #####

def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, :, C01:C02, C11:C12]


####### FFT Shifting #####
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    
####### Forward Model #####


def Hfor(model, x):
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    #X = torch.fft(batch_ifftshift2d(xc),2)
    X = torch.fft(xc,2)
    HX = complex_multiplication(model.H,X)
    out = torch.ifft(HX,2)
    out_r, _ = torch.unbind(out,-1)
    return out_r

def Hadj(model, x):
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    #X = torch.fft(batch_ifftshift2d(xc),2)
    X = torch.fft(xc,2)
    HX = complex_multiplication(model.Hconj,X)
    #out = batch_ifftshift2d(torch.ifft(HX,2))
    out = torch.ifft(HX,2)
    out_r, _ = torch.unbind(out,-1)
    return out_r


        
    
####### Soft Thresholding Functions  #####

def soft_2d_gradient2_rgb(model, v,h,tau):
    

    z0 = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    z1 = torch.zeros(model.batch_size, 3, 1, model.DIMS1*2, dtype = torch.float32, device=model.cuda_device)
    z2 = torch.zeros(model.batch_size, 3, model.DIMS0*2, 1, dtype= torch.float32, device=model.cuda_device)

    vv = torch.cat([v, z1] , 2)
    hh = torch.cat([h, z2] , 3)
    
    mag = torch.sqrt(vv*vv + hh*hh)
    magt = torch.max(mag - tau, z0, out=None)
    mag = torch.max(mag - tau, z0, out=None) + tau
    #smax = torch.nn.Softmax()
    #magt = smax(mag - tau, torch.zeros_like(mag, dtype = torch.float32))
    #mag = smax(mag - tau, torch.zeros_like(mag, dtype = torch.float32)) + tau
    mmult = magt/(mag)#+1e-5)
    if torch.any(mmult != mmult):
        print('here')
    if torch.any(v != v):
        print('there')

    return v*mmult[:,:, :-1,:], h*mmult[:,:, :,:-1]

def soft_2d(v,tau):
    out = torch.nn.functional.relu(v-tau)
    return out

######## normalize image #########
def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized


####### Add Noise #####
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise
    
######## ADMM Parameter Update #########
def param_update_previous(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    if s > res_tol*s:
        mu_up = mu/mu_dec
    else:
        mu_up = mu
   
    #mu_up = tf.cond(tf.greater(r, res_tol * s), lambda: (mu * mu_inc), lambda: mu)
    #mu_up = tf.cond(tf.greater(s, res_tol * r), lambda: (mu_up/mu_dec), lambda: mu_up)
    
    return mu_up

######## ADMM Parameter Update #########
def param_update2(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    else:
        mu_up = mu
        
    if s > res_tol*r:
        mu_up = mu_up/mu_dec
    else:
        mu_up = mu_up
   
    #mu_up = tf.cond(tf.greater(r, res_tol * s), lambda: (mu * mu_inc), lambda: mu)
    #mu_up = tf.cond(tf.greater(s, res_tol * r), lambda: (mu_up/mu_dec), lambda: mu_up)
    
    return mu_up
    
###### Things I saw on TV ###########
def make_laplacian(model):
    lapl = np.zeros([model.DIMS0*2,model.DIMS1*2])
    lapl[0,0] =4.; 
    lapl[0,1] = -1.; lapl[1,0] = -1.; 
    lapl[0,-1] = -1.; lapl[-1,0] = -1.; 

    LTL = np.abs(np.fft.fft2(lapl))
    return LTL


#def DT(dx, dy):  # Use convolution instead?  
#        with tf.device("/cpu:0"):
#            out = (tf.manip.roll(dx, 1, axis = 1) - dx) + (tf.manip.roll(dy, 1, axis = 2) - dy)
#        return out

#def D(x):
#    with tf.device("/cpu:0"):
#        xroll = tf.manip.roll(x, -1, axis = 1)
#        yroll = tf.manip.roll(x, -1, axis = 2)
#    return (xroll - x), (yroll - x)
    
    
def L_tf(a): # Not using
    xdiff = a[:,:, 1:, :]-a[:,:, :-1, :]
    ydiff = a[:,:, :, 1:]-a[:,:, :, :-1]
    return -xdiff, -ydiff

def Ltv_tf(a, b): # Not using
    return torch.cat([a[:,:, 0:1,:], a[:,:, 1:, :]-a[:,:, :-1, :], -a[:,:,-1:,:]],
                2) + torch.cat([b[:,:,:,0:1], b[:, :, :, 1:]-b[:, :, :,  :-1], -b[:,:, :,-1:]],3)
    #return tf.concat([a[:,0:1,:], a[:, 1:, :]-a[:, :-1, :], -a[:,-1:,:]], axis = 1) + tf.concat([b[:,:,0:1], b[:, :, 1:]-b[:, :,  :-1], -b[:,:,-1:]], axis = 2)


def TVnorm_tf(x):
    x_diff, y_diff = L_tf(x)
    result = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
    return result
