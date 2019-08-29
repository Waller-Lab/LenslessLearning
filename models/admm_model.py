import torch
import torch.nn as nn
from admm_helper_functions_torch import *
from admm_rgb_pytorch import *
import admm_filters_no_soft as admm_s

class ADMM_Net(nn.Module):
    def __init__(self, batch_size, h, iterations, learning_options = {'learned_vars': []}, 
                 cuda_device = torch.device('cpu'), le_admm_s = False, denoise_model = []):
        super(ADMM_Net, self).__init__()
        
        self.iterations = iterations              # Number of unrolled iterations
        self.batch_size = batch_size              # Batch size 
        self.autotune = False                     # Using autotune (True or False)
        self.realdata = True                      # Real Data or Simulated Measurements
        self.printstats = False                   # Print ADMM Variables
        
        self.addnoise = False                     # Add noise (only if using simulated data)
        self.noise_std = 0.05                     # Noise standard deviation 
        self.cuda_device = cuda_device
        
        self.l_admm_s = le_admm_s                      # Turn on if using Le-ADMM*, otherwise should be set to False
        if le_admm_s == True:
            self.Denoiser = denoise_model.to(cuda_device)
        
        # Leared structure options   
        self.learning_options = learning_options
        print(learning_options['learned_vars'])

        
        ## Initialize constants 
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions
        
        self.PAD_SIZE0 = int((self.DIMS0)//2)                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)                           # Pad size
        
        # Initialize Variables 
        self.initialize_learned_variables(learning_options)
        

        # PSF
        self.h_var = torch.nn.Parameter(torch.tensor(h, dtype=torch.float32, device=self.cuda_device),
                                            requires_grad=False)
            
        self.h_zeros = torch.nn.Parameter(torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float32, device=self.cuda_device),
                                          requires_grad=False)

        
        self.h_complex = torch.stack((pad_zeros_torch(self, self.h_var), self.h_zeros),2).unsqueeze(0)
        
        self.H = torch.fft(batch_ifftshift2d(self.h_complex).squeeze(), 2)   
        self.Hconj =  self.H* torch.tensor([1,-1], dtype = torch.float32, device=self.cuda_device) 
        self.HtH = complex_abs(complex_multiplication(self.H, self.Hconj))
         

        self.LtL = torch.nn.Parameter(torch.tensor(make_laplacian(self), dtype=torch.float32, device=self.cuda_device),
                                      requires_grad=False)
        
        self.resid_tol =  torch.tensor(1.5, dtype= torch.float32, device=self.cuda_device)
        self.mu_inc = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device)
        self.mu_dec = torch.tensor(1.2, dtype = torch.float32, device=self.cuda_device) 

    def initialize_learned_variables(self, learning_options):
        
        if 'mus' in learning_options['learned_vars']:  # Make mu parameters learnable
            self.mu1= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32))
            self.mu2= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32))
            self.mu3= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*1e-4, dtype = torch.float32))
        else:                                          # Not learnable
            self.mu1=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
            self.mu2=  torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4
            self.mu3 = torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*1e-4


        if 'tau' in learning_options['learned_vars']:  # Make tau parameter learnable
            self.tau= torch.nn.Parameter(torch.tensor(np.ones(self.iterations)*2e-4, dtype = torch.float32))
            
        else:
            self.tau= torch.ones(self.iterations, dtype = torch.float32, device=self.cuda_device)*2e-3 
    


    def forward(self, inputs):    
        
        self.batch_size = inputs.shape[0]

        #self.HtH = complex_abs(complex_multiplication(self.H, self.Hconj))
        
        self.mu_vals = torch.stack([self.mu1, self.mu2, self.mu3, self.tau])
        
        self.admmstats = {'dual_res_s': [], 'dual_res_u': [], 'dual_res_w': [], 
             'primal_res_s': [], 'primal_res_u': [], 'primal_res_w': [],
             'data_loss': [], 'total_loss': []}
        
        if self.autotune==True:
            self.mu_auto_list = {'mu1': [], 'mu2': [], 'mu3': []}

        # If using simulated data, input the raw image and run through forward model
        if self.realdata == False: 
            y = crop(self, self.Hfor(pad_dim2(self, inputs)))
            if self.addnoise == True:
                y = self.gaussian_noise_layer(y, self.noise_std)
        
        
        # Otherwise, input is the normalized Diffuser Image 
        else:
            y = inputs
        
            
        Cty = pad_zeros_torch(self, y)                      # Zero padded input
        CtC = pad_zeros_torch(self, torch.ones_like(y))     # Zero padded ones 
        
        # Create list of inputs/outputs         
        in_vars = []; in_vars1 = []
        in_vars2 = []; Hsk_list = []
        a2k_1_list=[]; a2k_2_list= []

        sk = torch.zeros_like(Cty, dtype = torch.float32)
        alpha1k = torch.zeros_like(Cty, dtype = torch.float32)
        alpha3k = torch.zeros_like(Cty, dtype = torch.float32)
        Hskp = torch.zeros_like(Cty, dtype = torch.float32)

        if self.l_admm_s == True:
            Lsk_init, mem_init = self.Denoiser.forward(sk)
            alpha2k = torch.zeros_like(Lsk_init, dtype = torch.float32,  device=self.cuda_device)  
        else:
        
            alpha2k_1 = torch.zeros_like(sk[:,:,:-1,:], dtype = torch.float32)  
            alpha2k_2 = torch.zeros_like(sk[:,:,:,:-1], dtype = torch.float32)
            
            a2k_1_list.append(alpha2k_1)
            a2k_2_list.append(alpha2k_2)

        mu_auto = torch.stack([self.mu1[0], self.mu2[0], self.mu3[0], self.tau[0]])

        
        in_vars.append(torch.stack([sk, alpha1k, alpha3k, Hskp]))
        

        

        for i in range(0,self.iterations):
            
            if self.l_admm_s == True:
                
                out_vars, alpha2k, _ , symm, admmstats = admm_s.admm(self, in_vars[-1], alpha2k, CtC, Cty, [], i, y)
                in_vars.append(out_vars)
                
            else:
            
                if self.autotune==True:
                    out_vars, a_out1, a_out2, mu_auto , symm, admmstats= admm(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, mu_auto, i, y)

                    self.mu_auto_list['mu1'].append(mu_auto[0])
                    self.mu_auto_list['mu2'].append(mu_auto[1])
                    self.mu_auto_list['mu3'].append(mu_auto[2])

                else:
                    out_vars, a_out1, a_out2, _ , symm, admmstats = admm(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, [], i, y)

                #if torch.any(out_vars != out_vars):
                #    print('loop')

                in_vars.append(out_vars)
                a2k_1_list.append(a_out1)
                a2k_2_list.append(a_out2)


            if self.printstats == True:                   # Print ADMM Variables
                self.admmstats['dual_res_s'].append(admmstats['dual_res_s'])
                self.admmstats['primal_res_s'].append(admmstats['primal_res_s'])
                self.admmstats['dual_res_w'].append(admmstats['dual_res_w'])
                self.admmstats['primal_res_w'].append(admmstats['primal_res_w'])
                self.admmstats['dual_res_u'].append(admmstats['dual_res_u'])
                self.admmstats['primal_res_u'].append(admmstats['primal_res_u'])
                self.admmstats['data_loss'].append(admmstats['data_loss'])
                self.admmstats['total_loss'].append(admmstats['total_loss'])
                
                
            x_out = crop(self, in_vars[-1][0])
            x_outn = normalize_image(x_out)
            self.in_list = in_vars
            
        return x_outn#, symm

        
