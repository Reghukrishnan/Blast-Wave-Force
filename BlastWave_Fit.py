# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:20:03 2022

@author: Reghukrishnan G

#: table_doi: 10.17182/hepdata.61925.v1/t7
#: name: Table 7
#: description: pT-differential invariant yield of pion+ and pion- for centrality 50-60%.

#: data_file: Table_7.yaml
#: keyword reactions: PB PB --> PI+ X | PB PB --> PI- X
#: keyword observables: DN/DPT
#: keyword phrases: Inclusive | Differential Distribution | Transverse Momentum Dependence
#: keyword cmenergies: 2760.0
#: CENTRALITY [pct],,,50.0-60.0
#: RE,,,PB PB --> PI+ X
#: SQRT(S)/NUCLEON [GeV],,,2760.0
#: YRAP,,,-0.5-0.5
PT [GEV],PT [GEV] LOW,PT [GEV] HIGH,(1/Nev)*(1/(2*PI*PT))*D2(N)/DPT/DYRAP [GEV**-2],stat +,stat -,sys +,sys -,"sys,normalization uncertainty +","sys,normalization uncertainty -"
"""

from scipy.special import i0,k1,kv
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




#m =  469/500 #P
#m = 0.13957039  #PI

#m = 0.493677 #K
#mp2 = m*m


Pi = np.pi
Sqrt = np.sqrt
Sinh = np.sinh
Cosh = np.cosh



@lru_cache(maxsize=64)
def BIfN(pT,u0,N):
    
    mT = Sqrt(mp2 + pT*pT)
    xi = m/T
    
    
    res = integrate.quad(lambda r : r*i0(pT*ur(u0,r)/T)*k1(mT*ut(u0,r)/T),0,1)
    return N*mT*res[0]/(T*kv(2,xi))
    

@lru_cache(maxsize=64)
def ur(u0,r,n=1):
    return u0*r**n#/Sqrt(1 -(u0*r**n)**2)
    
@lru_cache(maxsize=64)
def ut(u0,r,n=1):
    return Sqrt(1 +(u0*r**n)**2)

# Calculates BetaT average
def bTavg(x):
    sx = Sqrt(x)
    return 2*x*(Sqrt(1 + x)/x - Sinh(sx)/(x*sx))

# Calculates BetaT max    
def bTmax(x): 
    return x/Sqrt(1+x*x)

def chiSq(obs,exp,ebar):
    return np.sum(((exp-obs)/ebar)**2)/3



BIfN = np.vectorize(BIfN)

Particles = {"P":0.938,"PBar":0.938,"PI":0.13957039,"PIBar":0.13957039,"K":0.493677,"KBar":0.493677}

Particle = namedtuple('Particle',['mass','T','u0','A','bTavg','bTmax','chisq'])

Particle_Data = {}

LateX_Table_begin0 = r"\begin{tabular}{| l | c| c | c|} \hline " 
LateX_Table_begin1 = r"Particle              & $\left \langle \beta_T \right \rangle$ & $\beta_{max}$  & $\chi^2_{red}$\\[0.5ex] \hline"


"""   -----------  CONTROLS  ----------              """

cent ='56' ##05,23,34,56

T = 0.13



"""---------------------------------------------- """


with open("Results/ResultTex/BWFit{}.txt".format(cent),'w') as f:         # Opening a file to store data in LateX format
    
    #--------------------------------------------
    f.write(r"\begin{center}")                  #
    f.write('\n')                               #
    f.write(LateX_Table_begin0)                 #
    f.write('\n')                               #
    f.write(LateX_Table_begin1)                 #
    #--------------------------------------------
    
    for particle,m in Particles.items():    
        
        mp2 = m*m
        
        ###             Taking Data as input from a csv file            ###
        Col_data = np.genfromtxt('Data/Data{}/{}.csv'.format(cent,particle + cent ), delimiter=',')
        
        
        
        l =6                        # initial data point
        h = -1                      # Final data point
        px = Col_data[l:h,0]
        
        py = Col_data[l:h,3]
        
        ###             Experimental Error - Statistical + Systemic     ##
        sigma = Col_data[l:h,4] + Col_data[l:h,6]    
        
        
        ##              Fittign Data            ##
        popt, pcov = curve_fit(BIfN,px,py,p0=[1.5,200],sigma=sigma,bounds=((0,0), (3,10000)),method = 'trf')
        
        print("------------------------------\n",particle)
        print("\n u0  =    " , popt[0],"\n")
        print(" Norm  =  " , popt[1],"\n")
        
        ##      Calculating y values with fitted parameters     ##
        pycal = BIfN(px,popt[0],popt[1])
        
        ##      Calculating Chi Square of the fit        ##
        chi = chiSq(py,pycal,sigma)
        print("\n chi^2 =",chi)
        print("\n bTavg =",bTavg(popt[0]))
        
    
        plt.figure()
    
        plt.xlabel(r"$p_T[GeV]$ ")
        plt.ylabel(r'$\frac{1}{2\pi N_ev}\frac{d^2N}{p_Tdp_Tdy}[Log c^2/Gev^2]$')
        
        
        ###-------------------------------------------------------------------###        
        if particle[-3:]=='Bar':
            
            yerr = np.log(py+sigma) -np.log(py)
            
            plt.plot(px,np.log(pycal), label=r'$\bar{}$ Fit'.format(particle[:-3]))  # \bar{P}
            plt.errorbar(px,np.log(py), label=r'$\bar{}$ Data'.format(particle[:-3]),yerr = yerr,ecolor = 'red',color='y')
            
            LateX_row = "$\\bar{%s}$"%particle[:-3] + r"   &   {:#.3g}  &   {:#.3g}   \\ [0.5ex] \hline ".format(bTavg(popt[0]),bTmax(popt[0]))
            
            f.write(LateX_row)
            f.write('\n')
        else:
            yerr = np.log(py+sigma) -np.log(py)
            
            plt.plot(px,np.log(pycal), label=r'${}$ Fit'.format(particle))  # \bar{P}
            plt.errorbar(px,np.log(py), label=r'${}$ Data'.format(particle),yerr = yerr,ecolor = 'red',color='y')
            
            LateX_row = r"${}$  &   {:#.3g}   &   {:#.3g}   \\ [0.5ex] \hline ".format(particle,bTavg(popt[0]),bTmax(popt[0]))
                       
            f.write(LateX_row)
            f.write('\n')
        
        # mass,u0,Temp,Normalisation,betaT average,betaTmax, chisquare
        Particle_Data[particle] = Particle(m,T,popt[0],popt[1],bTavg(popt[0]),bTmax(popt[0]),chi)._asdict() 
        ####-------------------------------------------------------------------###
        
        
        plt.title(r"Pb-Pb $\sqrt{{s}} = 2.76TeV$, Cent-{}".format(cent[0]+'0-'+cent[1]+'0'+'%'))
        plt.legend()
        plt.grid()
        plt.savefig('Graphs/Yield/{}.png'.format(particle+cent), bbox_inches="tight",dpi = 300)
    
    ###----------------------------------------------------------------------###              
    f.write(r"\end{tabular} ")
    f.write('\n')
    f.write(r"\end{center}")
    ###----------------------------------------------------------------------###

# Storing Fit results as a python dictionary
    
with open("Results/ResultsPyDict/Particle_Data{}.txt".format(cent),'w') as pdat:
    pdat.write(json.dumps(Particle_Data)) 
#