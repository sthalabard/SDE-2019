#!/home/sthalabard/anaconda3/bin/python
#!/impa/home/f/simon.thalabard/anaconda3/bin/python3

#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# A CLASS TO INTEGRATE THE STANDARD GOY AND SABRA MODELS IN VELOCITY VARIABLES
# INCLUDES A PARAMETER s:
#               s is the entropic dimension of our mode
#
# TO OBTAIN THE MODELS CONIDERED BY PROCACCIA AND AL:
#        -->MODIFY LINE 54 WITH gamma <- lambda**g anf g in[0,2]
#        ---> 
# ALSO INCLUDES A PARAMETER TO EXTEND THE INTEGRATION TO THE MODELS CONSIDERED BY
#--------------------------------#
from math import *
import numpy as np
import scipy as scp
import scipy.optimize 
from scipy import fftpack as fft
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from matplotlib import animation
from time import time as ctime
import os,glob

from . import Shell4KH_boost as boost

class GOY:
        def __init__(self,nu=1, hyper=1,d=2, s= 0., nmin=-12, nmax=10,Lambda=2, dt=1e-3,dtype='complex128'):

                #Parameters of the shell Model
                self.nu=nu      #Viscosity
                self.d=d        #Dimension
                self.nmin=nmin  #minimal shell number
                self.nmax=nmax  #Maximal shell number
                self.Lambda=Lambda #Intershell ratio

                self.n=nmax-nmin+1 #Number of shells
                self.dt=dt #timestep
                self.hyper=hyper #hyper viscosity
                self.k=np.arange(nmin,nmax+1); #wavenumber label
                self.s=s #ENTROPY of the shell model: set to 0 for standard GOY/SABRA set to d for relevant imitation of 2D dynamics
                self.eta=self.Lambda**(1.+self.s)                 

                self.a=np.zeros(self.n,dtype=dtype) #STATE (velocity)
                self.da=np.zeros(self.n,dtype=dtype) #DERIVATIVE
                self.forcing=np.zeros(self.n,dtype=self.a[0].real.dtype) #VARIANCE OF THE FORCING (WHITE NOISE)
                self.dtype=dtype
                
                self.a_xy=np.zeros((self.n,2),dtype=self.a[0].real.dtype) #REAL/IMAG VERSION OF a, to use in coordination with the boost library
                self.da_xy=np.zeros((self.n,2),dtype=self.a[0].real.dtype)

                self.time=0 #CURRENT TIME

                if d==2: self.gamma=Lambda**(2.)
                else:self.gamma=-Lambda 
                        
                self.Lambdas=self.Lambda**(1.*self.k)
                self.Gammas=self.gamma**(1.*self.k)
                self.Etas=self.eta**(1.*self.k)

                self.dissip= nu*self.Lambda**(2.* self.k *self.hyper)

                self.E=0. #!sum |a|^2
                self.Z=0.# sum \gamma |a|^2 

                #STAT
                self.S2=np.zeros_like(self.Gammas)
                self.S3=np.zeros_like(self.Gammas)
                self.S4=np.zeros_like(self.Gammas)

                self.update_laws()

        def print(self):
            print('nu =',self.nu)
            print('hyper =',self.hyper)
            print('d =',self.d)
            print('nnin =',self.nmin)
            print('nmax =',self.nmax)
            print('Lambda =',self.Lambda)
            print('Gamma/Lambda =', self.gamma/self.Lambda)
            print('s =',self.s)
            print('dt =',self.dt)

            
        def update_xy(self):
            self.a_xy[:,0],self.a_xy[:,1]=self.a.real[:],self.a.imag[:]
            self.da_xy[:,0],self.da_xy[:,1]=self.da.real[:],self.da.imag[:]

        def rhs_nl(self,a):
                out=np.zeros_like(a);
                astar=a.conjugate()
                out[:-2]=astar[1:-1]*astar[2:] *self.Lambda**2*self.gamma#a(k+1)*a(k+2)
                out[1:-1]+=astar[:-2]*astar[2:]*(-self.Lambda)*(1+self.gamma) #a(k-1)*a(k+1)
                out[2:]+=astar[:-2]*astar[1:-1] #a(k-2)a(k-1)
                out[:]*=self.Etas[:]
                return out
            
        def update_dt(self):
                a1=self.rhs_nl(self.a)
                a2=self.rhs_nl(self.a+0.5*self.dt*a1)
                a3=self.rhs_nl(self.a+0.5*self.dt*a2)
                a4=self.rhs_nl(self.a+self.dt*a3)
                self.da=(a1+2*a2+2*a3+a4)/6
                self.a=(self.a+self.dt*self.da)*np.exp(-self.dissip*self.dt)
                W1,W2=np.random.randn(self.n),np.random.randn(self.n)
                self.a[:]+=np.sqrt(2*fk*self.dt)*(W1[:]+1j*W2[:])

        def update(self,NT=10,use_boost=True):
                if use_boost:
                    self.update_xy()
                    boost.update(da=self.da_xy,a=self.a_xy,nt=NT,dt=self.dt,\
                                 f=self.forcing,dissip=self.dissip,\
                                 gam=self.gamma,lam=self.Lambda,etas=self.Etas)
                    self.a[:]=self.a_xy[:,0]+1j*self.a_xy[:,1]
                    self.da[:]=self.da_xy[:,0]+1j*self.da_xy[:,1]
                else:
                    for i in range(NT): self.update_dt()

                self.time=self.time+NT*self.dt
                self.update_laws()

        def update_laws(self):
                self.S2=(np.abs(self.a)**2)
                self.S3[1:-1]=2.*(self.a[:-2]*self.a[1:-1]*self.a[2:]).real 
                self.S4=np.sum(np.abs(self.a)**4)
                self.E=np.sum(np.abs(self.a)**2)
                self.Z=np.sum(np.abs(self.a)**2*self.Gammas)

####### Dynamique every : evolution lineaire is "every" is not "None"

        def evo(self,cmax=10,every=None,verbose=False,out=None):       
                a,da=np.zeros((self.n,cmax+1),dtype=self.dtype),np.zeros((self.n,cmax+1),dtype=self.dtype)
                t=np.zeros(cmax+1)
                t[0]=self.time
                compt=0
                tot=0
                a[:,compt]=self.a[:]
                da[:,compt]=self.da[:]
                tic0=ctime()
                for compt in range(cmax):
                        tic=ctime()
                        if every is None: NT=2**compt
                        else:NT=every
                        self.update(NT=NT,use_boost=True)
                        tot+=NT
                        a[:,compt+1]=self.a[:]
                        da[:,compt+1]=self.da[:]
                        t[compt+1]=self.time
                        if verbose:
                                print("%d \t NT=%d \t E = %0.8e \t Z= %0.8e \t %f" %(compt,NT,self.E,self.Z,ctime()-tic))
                print("%d Iterations in %0.2e s \t t=%f tau_Z \t\tE = %0.8e \t Z= %0.8e " %(tot, ctime()-tic0,self.time*np.abs(self.Z)**(0.5),self.E,self.Z))

                if out is not None:
                    np.savez(out,t=t,a=a,da=da,nu=self.nu,hyper=self.hyper,d=self.d,nmin=self.nmin, nmax=self.nmax,Lambda=self.Lambda,dt=self.dt,s=self.s)
                    print("evo saved to", out)
                return t,a,da

####### INITIALIZATION ROUTINES
        def init_KH(self,a=0,theta=pi/4):
            self.a[:]=0
            self.da[:]=0
            self.a[self.k%3==0]=(self.Lambdas[self.k%3==0]**(-a))*(np.cos(theta)+1j*np.sin(theta))
            self.update_laws()

        def regularize(self,kcut=None,tol=1e-30):
            if kcut is None:
                damp=0
            else:
                self.time=-0.5*np.log(tol)/self.dissip[self.k==kcut]
                damp=self.dissip*self.time;
            self.a=self.a*np.exp(-damp)
            self.update_laws()

        def init_Compact(self,k0=0,kf=2):
            self.a[:]=0
            self.da[:]=0
            n=(kf-k0+1)
            theta=np.random.rand(n)*2*pi
            self.a[(self.k<=kf)*(self.k>=k0)]=(np.cos(theta)+1j*np.sin(theta))/np.sqrt(2)
            self.update_laws()

        def init_with(self,a):
            self.a[:]=a[:]
            self.da[:]=0
            self.update_laws()

        def init_Gaussian(self,kf=2,sig=1):
            self.a[:]=0
            self.da[:]=0
            theta=np.random.rand(self.n)*2*pi
            self.a=np.exp(-(1.*self.k-kf)**2/sig)*(np.cos(theta)+1j*np.sin(theta))/np.sqrt(2)
            self.update_laws()

        def perturb(self,delta=0.1,c=0):
                W1,W2=np.random.randn(self.n),np.random.randn(self.n)
                self.a[:]+=np.sqrt(0.5*delta)*(1j*W2+W1)*np.sqrt(2*self.Lambdas[:]**(-c))

        def compute_spec(self):
            kpert=self.k[0:-2:3]
            Ek=np.abs(self.a)**2
            Ek_pert=(1*Ek[0:-2:3]+Ek[1:-1:3]+Ek[2::3])/3
            return self.k,kpert,Ek,Ek_pert

class ensemble:
    def __init__(self,IO=None,NB=None,prefix=''):
        if IO is None:
            print('must specify the IO! ')
        self.list=glob.glob(os.path.join(IO,prefix,'*.npz'))
        if len(self.list)==0:
            print('EMPTY ENSEMBLE !')
        else:
            tmp=np.load(self.list[0])
            self.nu=np.float(tmp['nu'])
            self.hyper=np.int(tmp['hyper'])
            self.d=np.int(tmp['d'])
            self.nmin=np.int(tmp['nmin'])
            self.nmax=np.int(tmp['nmax'])
            self.Lambda=np.float(tmp['Lambda'])
            self.dt=np.float(tmp['dt'])
            self.time=tmp['t']
            self.nt=len(self.time)
            self.s=np.float(tmp['s'])
            self.k=np.arange(self.nmin,self.nmax+1)
            self.n=len(self.k)
        
            if NB is not None:self.NB=min(NB,len(self.list))
            else:self.NB=len(self.list)

        flow=GOY(d=self.d,nu=self.nu,hyper=self.hyper,nmin=self.nmin,nmax=self.nmax,dt=self.dt,Lambda=self.Lambda,s=self.s)

        self.rhs_nl=flow.rhs_nl
        self.Gamma=flow.gamma
        self.Lambdas=flow.Lambdas
        self.Gammas=flow.Gammas
        self.Etas=flow.Etas
        
        #Energies, Enstrophies, and related anisotropic fluxes
        self.Ekt,self.PEkt,self.Zkt,self.PZkt = np.zeros_like(tmp['a']),np.zeros_like(tmp['a']),np.zeros_like(tmp['a']),np.zeros_like(tmp['a'])
        self.DZkt,self.DEkt = np.zeros_like(tmp['a']),np.zeros_like(tmp['a'])

        self.S2,self.S2dot,self.S3 = np.zeros_like(tmp['a']),np.zeros_like(tmp['a']),np.zeros_like(tmp['a'])

        print( 'number of realizations :' , self.NB)
        print('computing moments...')
        self.compute_av()
        print('done')

        ik0=np.argwhere(self.k%3==0)
        self.Eg=self.Ekt[ik0,:].sum(axis=0)
        self.Ep=(self.Ekt[ik0+2,:]+self.Ekt[ik0+1,:]).sum(axis=0)
        self.Etot=self.Ekt[:,:].sum(axis=0)

        self.Zg=self.Zkt[ik0,:].sum(axis=0)
        self.Zp=(self.Zkt[ik0+2,:]+self.Zkt[ik0+1,:]).sum(axis=0)
        self.Ztot=self.Zkt[:,:].sum(axis=0)

        self.tnu=self.time[0]
        self.TL=1./((self.Lambdas[0]**(1.5*self.s+1))*np.sqrt(self.Etot[0]))

    def compute_av(self):
        #Compute moments
        self.Ekt[:]=0;self.Zkt[:]=0;self.PEkt[:]=0
        self.S2[:]=0;self.S3[:]=0

        for i in range(self.NB):
            tmp=np.load(self.list[i])
            u=tmp['a']
            self.S2 += np.abs(u)**2/float(self.NB)
            self.S3[1:-1,:]+=2.*(u[0:-2,:]*u[1:-1,:]*u[2:,:]).real/float(self.NB)

        #Compute Fluxes and Spectrum
        for t in range(self.nt):
            self.PEkt[:-1,t]= 2*self.Lambdas[:-1]*(self.Lambda**(1)*self.S3[:-1,t]-self.Gamma*self.Lambda**(2)*self.S3[1:,t]).real
            self.PZkt[:-1,t]= 2*self.Gammas[:-1] * self.Lambdas[:-1]*(self.Lambda**(1)*self.S3[:-1,t]-self.Gamma*self.Lambda**(2)*self.S3[1:,t]).real

            self.DEkt[:,t]=2*np.cumsum((self.Lambdas[:]**(2*self.hyper-self.s))*self.S2[:,t])
            self.DZkt[:,t]=2*np.cumsum((self.Lambdas[:]**(2*self.hyper-self.s))*self.Gammas[:]*self.S2[:,t])

            self.Ekt[:,t]=self.S2[:,t]*(self.Lambdas[:]**(-self.s))
            self.Zkt[:,t]=self.S2[:,t]*(self.Lambdas[:]**(-self.s))*self.Gammas[:]
        return 1

if __name__== "__main__":

####### TESTS

    flow=GOY(d=2,nu=1,kdiss=16,hyper=1,nmin=-30,nmax=32,dt=1e-4,Lambda=2**(1./3),g=2)
    flow.init_Gaussian(sig=0.5,kf=0)  
    flow.f=flow.a.copy()
    flow.a[:]=0
    #flow.init_Compact(k0=-20,kf=-18)  
    t1,a1,da1=flow.evo(cmax=20)#LOG
