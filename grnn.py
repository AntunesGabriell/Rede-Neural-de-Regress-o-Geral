# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:42:36 2023

@author: gabri
"""

import numpy as np
from numba import jit, prange

class GRNN():
    def __init__(self, centros, weights_ys, gammas):
        
        self.centros= centros
        self.centros_lenght= len(centros)
        self.weights_ys= weights_ys
        self.weights_ys_lenght= len(weights_ys[0])        
        self.gammas= np.ones((1,self.centros_lenght))*gammas
        

    def forward(self, previsores):
        print(len(previsores))
        dist= calcula_distancia(centros= self.centros,  n_centros= self.centros_lenght,
                              base= previsores, base_lenght= len(previsores),
                              gammas= self.gammas)
        
        fi= self.aplica_gaussiana(dist)
        fi= np.ascontiguousarray(fi)
        resultado= result(fi= fi, fi_lenght= len(previsores), ys_centros=self.weights_ys, 
                          tam_ys_centros= self.weights_ys_lenght  )
        
        return resultado

    def aplica_gaussiana(self, distancias):
        return 2.71828**(-1/(2*self.gammas**2)* distancias)

@jit(nopython=True, parallel=True)
def calcula_distancia(centros, n_centros, base, base_lenght, gammas):
    
    dist= np.ones((base_lenght, n_centros))    
    # entrada - centros
    for i in prange(base_lenght):      
        for j in prange(n_centros):    
            d= centros[j]- base[i]
            dist[i,j]=  np.dot( d ,  d.T)       
            
    return dist


@jit(nopython=True, parallel=True)
def result( fi, fi_lenght , ys_centros, tam_ys_centros): 
    
    resultado= np.ascontiguousarray( np.ones((fi_lenght,tam_ys_centros)))
    
    for i in prange(fi_lenght):       
        for j in range(tam_ys_centros):           
            numerador= np.dot( fi[i] , ys_centros[:,j])
            denominador=  np.sum(fi[i])
            resultado[i,j]=numerador/denominador
        
    
    return resultado


def xor():
    xor= np.array([[1,1,1],[0,1,0],[0,0,1],[1,0,0]], dtype= np.float64)
   
    grnn=  GRNN(centros= xor[:,:2], weights_ys= xor[:2,2:], gammas= 0.1)
    resultado= grnn.forward( previsores= xor[2:,:2])
    
    print(resultado)

   

if __name__ == '__main__':
    xor()

















