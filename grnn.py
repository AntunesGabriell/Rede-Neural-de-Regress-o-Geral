# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:42:36 2023

@author: gabri
"""

import numpy as np
from numba import jit, prange
import random as rn
from copy import deepcopy

class GRNN():
    def __init__(self, centros, weights_ys, gammas):
        
        self.centros= centros
        self.centros_lenght= len(centros)
        self.weights_ys= weights_ys
        self.weights_ys_lenght= len(weights_ys[0])        
        self.gammas= np.ones((1,self.centros_lenght))*gammas
        
        

    def forward(self, previsores, gammas= None):
        
        dist= calcula_distancia(centros= self.centros,  n_centros= self.centros_lenght,
                            base= previsores, base_lenght= len(previsores))
        
       
        if gammas is None :          
            
            fi= self.aplica_gaussiana(dist, self.gammas)
            
        else:     
            
            fi= self.aplica_gaussiana(dist, gammas)
        
        
        fi= np.ascontiguousarray(fi)
        resultado= result(fi= fi, fi_lenght= len(previsores), ys_centros=self.weights_ys, 
                          tam_ys_centros= self.weights_ys_lenght  )
        
        return resultado

    def aplica_gaussiana(self, distancias, gammas):
        
        return 2.71828**(-1/(2*gammas**2)* distancias)
    
    
    def E_D(self, tam_pop, largura_media, previsores_train, y_train,
            n_geracoes, f_escala=0.2, prop_mut=0.3):

        # gera população inicial
        gammas= np.random.normal(loc=largura_media , scale= 0.2, 
                                 size=(tam_pop, self.centros_lenght)).round(2)
        
        fitnes_pop= []
        # calcula erro da pop inicial
        for i in range(tam_pop):           
            a= self.forward(previsores= previsores_train, gammas= gammas[i])
            erro= self.mean_absolute_error(real= y_train, previsto=a)
            fitnes_pop.append(erro)
           
        
        for i in range(n_geracoes):  
            new_gammas= []
            fitnes_new_popy=[]
            print ('\r',f'Geração: {i}', end='', flush=True)
            
            for j in range(tam_pop):

                #mutacao
                indices= rn.sample(range(0,tam_pop), 3)
                u_g= gammas[indices[0]]+ f_escala*(gammas[indices[1]]-gammas[indices[2]])

                #cruzamento
                for k in range(self.centros_lenght):
                    if rn.random()>prop_mut:                    
                        u_g[k]= gammas[j][k] 
                
                u_g+=0.001
                a= self.forward(previsores= previsores_train, gammas=u_g )
                fitnes_u_g= self.mean_absolute_error(real= a, previsto= y_train)
                
                fitnes_x= fitnes_pop[j]
                #selecao
                
                if fitnes_u_g<fitnes_x:
                    
                    fitnes_new_popy.append( fitnes_u_g)
                    new_gammas.append( u_g)
                else:               
                    fitnes_new_popy.append( fitnes_x)
                    new_gammas.append( gammas[j])
             
            
            gammas= deepcopy(new_gammas)
            fitnes_pop= deepcopy(fitnes_new_popy)
            menor= float ('inf')
       
        indice=0
        for i in range(tam_pop):        
            if menor > fitnes_pop[i]:
                menor= fitnes_pop[i]
                indice=i
                
        
       
        return gammas[indice]

    def mean_absolute_error(self, real, previsto):    
        return abs(real-previsto).sum()/len(real)
        
        
        

@jit(nopython=True, parallel=True)
def calcula_distancia(centros, n_centros, base, base_lenght):
    
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



  
if __name__ == '__main__':
    xor= np.array([[1,1,1],[0,1,0],[0,0,1],[1,0,0]], dtype= np.float64)
   
    grnn=  GRNN(centros= xor[:,:2], weights_ys= xor[:,2:], gammas= 0.1)
    gammas= grnn.E_D(tam_pop=10, largura_media=0.7, previsores_train= xor[:,:2], 
                     y_train= xor[:,2:], n_geracoes= 10)
    resultado= grnn.forward( previsores= xor[2:,:2], gammas= gammas)
    
    print(resultado)
















