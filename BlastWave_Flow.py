# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:15:05 2022

@author: Reghukrishnan G
"""

from scipy.special import i0,k1,kv,iv
import scipy.integrate as integrate
from scipy.optimize import curve_fit,minimize,leastsq
import numpy as np


from matplotlib import pyplot as plt
from matplotlib import rc

import json
from collections import namedtuple
from functools import lru_cache


font = {'family' : 'sans-serif',
        'weight' : 2,
        'size'   : 10}

rc('font', **font)



Pi = np.pi
Sqrt = np.sqrt
Sinh = np.sinh
Cosh = np.cosh
Cos  = np.cos
Sum  = np.sum


"""
 ###################### Function Definitions
"""
        
        
@lru_cache(maxsize=128)
def ur(r,phi,u0 ,c2 ): 
    
    return u0*r*(1 + c2*Cos(2*phi))

@lru_cache(maxsize=128)
def ut(r,phi,u0 ,c2 ):
    
    return Sqrt(1 + ur(r,phi,u0,c2)**2)


@lru_cache(maxsize=128)
def dN(pT,phi,prl,c2):

    m,u0,T = prl     
    mT = Sqrt(m**2+ pT*pT)        
    
    res = integrate.quad(lambda r : r*i0(pT*ur(r,phi,u0,c2)/T)*k1(mT*ut(r,phi,u0,c2)/T),0,1)
    
    return res[0]

@lru_cache(maxsize=128)
def dNv2(pT,phi,prl,c2):

    m,u0,T = prl     
    mT = Sqrt(m**2+ pT*pT)  
    
    res = integrate.quad(lambda r : r*iv(2,pT*ur(r,phi,u0,c2)/T)*k1(mT*ut(r,phi,u0,c2)/T),0,1)
    
    return res[0]

@lru_cache(maxsize=64)
def INphi(pT,prl,c2):
    return integrate.quad(lambda phi: dN(pT,phi,prl,c2), 0, 2*Pi)[0]

#INphi = np.vectorize(INphi,excluded=['prl'])

def v2(pT,m=0.13957039,u0 = 1.41,c2 =0.3):
    
    prl = (m,u0,T)
    return integrate.quad(lambda phi:Cos(2*phi)*dNv2(pT,phi,prl,c2), 0, 2*Pi)[0]/INphi(pT,prl,c2)


v2 = np.vectorize(v2)


"""   -----------  CONTROLS  ----------                     """
                                                                #
                                                                #                                                                            
T       = 0.13        #Temperature in GeV                       #
                                                                #
cent    = '05'        #Centrality 56 -> 50-60% ##05,23,34,56    #
                                                                #  
"""---------------------------------------------------------"""


"""
 ###################### Initialising and Loading Data
"""

Particle = namedtuple('Particle',['mass','T','u0','A','bTavg','bTmax','chisq'])

Particles = {}

with open('Results/ResultsPyDict/Particle_Data{}.txt'.format(cent)) as f:
    Part =  json.load(f)
    for key,value in Part.items():
        Particles[key] = Particle(**value)


#------------------------------------------------------------------------------
Particle_V2Data = {}

for Pname,P in Particles.items():
    
    if Pname[-3:]!='Bar':
        PartDat =np.genfromtxt("Data/Data{}/v2{}{}.csv".format(cent,Pname,cent),delimiter =',')
        
        l=0
        h = -12
        px =PartDat[:h,0]
        vy =PartDat[:h,1]
        sigma = PartDat[l:h,2] + PartDat[l:h,5]
        
        
        
        """
         ################### Fitting Data and Ploting
         
        """
        popt,pcov =curve_fit(lambda pT,c2 : v2(pT,m=P.mass,u0=P.u0,c2=c2),px,vy,p0=[0.1],sigma=sigma,bounds=((0), (10)),method = 'trf')
        
        
        vycal = v2(px,c2=popt[0])
        chi = np.sum(((vy-vycal)/sigma)**2)
        
        
        Particle_V2Data[Pname] = {'m':P.mass,'T':T,'u0':P.u0,'A':P.A,'bTavg':P.bTavg,"bTmax":P.bTmax,'chiYield':P.chisq,'c2':popt[0],'chiV2':chi}
        
        plt.figure()    
        plt.xlabel(r'$p_T$(GeV)')
        plt.ylabel(r"$v_2$ ")
        
        plt.plot(px,vycal,label=r'$v_2$ Fit')
        plt.errorbar(px,vy,yerr = sigma,ecolor = 'red',color='yellow',label=r'$v_2$ Data',barsabove =True,marker ="x")
        
        plt.title(r"Pb-Pb $\sqrt{{s}} = 2.76TeV$ -Charged {}".format(Pname))
        plt.legend()
        plt.grid()
        plt.savefig('Graphs/V2/{}.png'.format(Pname+cent), bbox_inches="tight",dpi = 300)
        
        print("\n Particle    : ",Pname)
        print("\n c2          : " ,popt[0])       
        print("\n chi squared : ",chi)
        print("----------------------------")

with open("Results/ResultsPyDict/Particle_V2Data{}.txt".format(cent),'w') as pdat:
    pdat.write(json.dumps(Particle_V2Data))