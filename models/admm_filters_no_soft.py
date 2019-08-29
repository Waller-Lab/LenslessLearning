#import tensorflow as tf
import numpy as np 
#import tensorflow_probability as tfp
import torch
from admm_helper_functions_torch import *
#from networks import *
import time

#def admm_rgb(model, in_vars, n, CtC, Cty, mu_auto, i):
def admm(model, in_vars, alpha2k, CtC, Cty, mu_auto, n, y):  
    
    sk = in_vars[0];  alpha1k = in_vars[1]; #alpha2k = in_vars[2];  
    alpha3k = in_vars[2]
    Hskp = in_vars[3]; 
    
    #alpha2k_1 = in_vars[4];  alpha2k_2 = in_vars[5]
    
    if model.autotune == True:
        mu1 = mu_auto[0];  mu2 = mu_auto[1];  mu3 = mu_auto[2]

    else:
        #mu1 = model.mu_vals[0][n];  mu2 = model.mu_vals[1][n];  mu3 = model.mu_vals[2][n]
        mu1 = model.mu1[n];  mu2 = model.mu2[n];  mu3 = model.mu3[n]
        
    tau = model.tau[n] #model.mu_vals[3][n]
    
    dual_resid_s = [];  primal_resid_s = []
    dual_resid_u = [];  primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []

    #Smult = 1/(mu1*model.HtH + mu2*model.LtL + mu3)  # May need to expand dimensions 
    Smult = 1/(mu1*model.HtH + mu2 + mu3)  # May need to expand dimensions 
    Vmult = 1/(CtC + mu1)
    
    ###############  update u = soft(Ψ*x + η/μ2,  tau/μ2) ###################################
    #Lsk1, Lsk2 = L_tf(sk)        # X and Y Image gradients 
    #Lsk, mem = model.Denoiser.forward(sk)

    

    #ukp = model.Denoiser(sk + alpha2k/mu2) + sk
    #Lsk = ukp
    #print(alpha2k.shape, Lsk.shape)
    #ukp = soft_2d(Lsk+alpha2k/mu2,tau) 
    
    tt1 = time.time()
    net_out, mem = model.Denoiser.forward(sk)
    tt2 = time.time()
    #print('inner', tt2-tt1)
    
    #ukp = soft_2d(Lsk+mu2,tau)
    #ukp_1, ukp_2 = soft_2d_gradient2_rgb(model, Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau)
    
    ################  update      ######################################

    vkp = Vmult*(mu1*(alpha1k/mu1 + Hskp) + Cty)

    ################  update w <-- max(alpha3/mu3 + sk, 0) ######################################


    zero_cuda = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
        
    wkp = torch.max(alpha3k/mu3 + sk, zero_cuda, out=None)
   


    skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj(model, vkp - alpha1k/mu1) + mu2*net_out
    
    #model.Denoiser.inverse(ukp - alpha2k/mu2, mem) 
    
    #skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj(model, vkp - alpha1k/mu1) + mu2*(ukp - alpha2k/mu2) 
    #symm = torch.sum(torch.abs(model.Denoiser.inverse(Lsk, mem) - sk))
    symm = []
        

    SKP_numerator = torch.fft(make_complex(skp_numerator), 2)
    skp = make_real(torch.ifft(complex_multiplication(make_complex(Smult), SKP_numerator), 2))
    
    Hskp_up = Hfor(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * torch.norm(Hskp - Hskp_up))
    primal_resid_s.append(torch.norm(r_sv))

    # Autotune
    if model.autotune == True:
        mu1_up = param_update2(mu1, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_s[-1], dual_resid_s[-1])
        #model.mu_vals[0][n+1] = model.mu_vals[0][n+1] + mu1_up
    else: 
        if n == model.iterations-1:
            mu1_up = model.mu_vals[0][n]
        else:
            mu1_up = model.mu_vals[0][n+1]

    alpha1kup = alpha1k + mu1*r_sv

    #Lskp, _ = model.Denoiser.forward(skp)

    #Lskp1, Lskp2 = L_tf(skp)
    #r_su = Lskp - ukp
    #r_su_1 = Lskp1 - ukp_1
    #r_su_2 = Lskp2 - ukp_2

    #dual_resid_u.append(mu2*torch.norm(Lsk - Lskp))
    #primal_resid_u.append(torch.norm(r_su))

    #if model.autotune == True:
    #    mu2_up = param_update2(mu2, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_u[-1], dual_resid_u[-1])
    #else:
    #    if n == model.iterations-1:
    #        mu2_up = model.mu_vals[1][n]
    #    else:
    #        mu2_up = model.mu_vals[1][n+1]

    alpha2kup = alpha2k #+ mu2*r_su
    #alpha2k_1up= alpha2k_1 + mu2*r_su_1
    #alpha2k_2up= alpha2k_2 + mu2*r_su_2
    mu2_up = mu2

    r_sw = skp - wkp
    dual_resid_w.append(mu3*torch.norm(sk - skp))
    primal_resid_w.append(torch.norm(r_sw))

    if model.autotune == True:
        mu3_up = param_update2(mu3, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_w[-1], dual_resid_w[-1])
    else:
        if n == model.iterations-1:
            mu3_up = model.mu_vals[2][n]
        else:
            mu3_up = model.mu_vals[2][n+1]

    alpha3kup = alpha3k + mu3*r_sw

    #Smult_up = 1/(mu1_up*model.HtH + mu2_up*model.LtL + mu3_up)
    #Vmult_up = 1/(model.CtC + mu1_up)

    #sk = skp
    #cost.append(tf.linalg.norm(model.crop(Hskp_up)-y)**2)
    data_loss = torch.norm(crop(model, Hskp_up)-y)**2
    tv_loss = tau*TVnorm_tf(skp)

    
    if model.printstats == True:
        
        admmstats = {'dual_res_s': dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_s':  primal_resid_s[-1].cpu().detach().numpy(),
                     'dual_res_w':dual_resid_w[-1].cpu().detach().numpy(),
                     'primal_res_w':primal_resid_w[-1].cpu().detach().numpy(),
                     'dual_res_u':dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_u':primal_resid_s[-1].cpu().detach().numpy(),
                     'data_loss':data_loss.cpu().detach().numpy(),
                     'total_loss':(data_loss+tv_loss).cpu().detach().numpy()}
        
        
        print('\r',  'iter:', n,'s:', admmstats['dual_res_s'], admmstats['primal_res_s'], 
         'u:', admmstats['dual_res_u'], admmstats['primal_res_u'],
          'w:', admmstats['dual_res_w'], admmstats['primal_res_w'], end='')
    else:
        admmstats = []

    

    #out_vars = torch.stack([skp, alpha1kup, alpha2kup, alpha3kup, Hskp_up])
    out_vars = torch.stack([skp, alpha1kup, alpha3kup, Hskp_up])

 
    mu_auto_up = torch.stack([mu1_up, mu2_up, mu3_up])
    
    return out_vars, alpha2kup, mu_auto_up, symm, admmstats
    
    
