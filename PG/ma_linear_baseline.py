#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:41:46 2018

@author: nehachoudhary
"""

import numpy as np
import copy
#should 'o' have input dimensions the size of self.n in MLP?
class LinearBaseline:
    def __init__(self, N, reg_coeff=1e-5):
        self.N = N
        self.n = N+1
        self._reg_coeff = reg_coeff
        self._coeffs = None
        self.variables = [i for i in range(self.N)]

    def _features(self, path):
        # compute regression features for the path
        feat = {}
        for i in range(self.N):
            ##second obs onwards [1:] - resolved
            
            #path["o"] or path["observations"]
            o = np.clip(path["o"][self.variables[i]], -10, 10)
            #o = np.clip(path["observations][self.variables[i]], -10, 10)
            #reshape even for ndim <=2
            o = o.reshape(o.shape[0], -1)
            if o.ndim > 2:
                o = o.reshape(o.shape[0], -1)
            l = len(path["rewards"][self.variables[i]])
            al = np.arange(l).reshape(-1, 1) / 1000.0
            feat[self.variables[i]] = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)            
        return feat
    
    '''
    def _features(path):
        # compute regression features for the path
        feat = {}
        for i in range(N):
            ##second obs onwards [1:] - resolved
            o = np.clip(path["observations"][variables[i]], -10, 10)
            #reshape even for ndim <=2
            o = o.reshape(o.shape[0], -1)
            if o.ndim > 2:
                o = o.reshape(o.shape[0], -1)
            l = len(path["rewards"][variables[i]])
            al = np.arange(l).reshape(-1, 1) / 1000.0
            feat[variables[i]] = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)            
        return feat
    '''

    def fit(self, paths, return_errors=False):
        
        featmat = {}
        for i in range(self.N):
            featmat[self.variables[i]] = np.concatenate([self._features(path)[self.variables[i]] for path in paths]) #add self._feat
        returns = {}
        for i in range(self.N):
            returns[self.variables[i]] = np.concatenate([path["returns"][self.variables[i]] for path in paths])

        if return_errors:
            predictions = {}
            errors = {}
            error_before = {}
            for i in range(self.N):
                predictions[self.variables[i]] = featmat[self.variables[i]].dot(self._coeffs) if self._coeffs is not None else np.zeros(returns[self.variables[i]].shape)
                #predictions[self.variables[i]] = featmat[self.variables[i]].dot(_coeffs) if _coeffs is not None else np.zeros(returns[self.variables[i]].shape)
                errors[self.variables[i]] = returns[self.variables[i]] - predictions[self.variables[i]]
                error_before[self.variables[i]] = np.sum(errors[self.variables[i]]**2)/np.sum(returns[self.variables[i]]**2)

        reg_coeff_init = copy.deepcopy(self._reg_coeff)
        #reg_coeff_init = copy.deepcopy(_reg_coeff)
        reg_coeff = {}
        _coeffs = {}
        for i in range(self.N):
            reg_coeff[self.variables[i]] = reg_coeff_init
            for _ in range(10):
                #self._coeffs = np.linalg.lstsq(
                _coeffs[self.variables[i]] = np.linalg.lstsq(
                    featmat[self.variables[i]].T.dot(featmat[self.variables[i]]) + reg_coeff[self.variables[i]] * np.identity(featmat[self.variables[i]].shape[1]),
                    featmat[self.variables[i]].T.dot(returns[self.variables[i]])
                )[0]
                #if not np.any(np.isnan(self._coeffs)):
                if not np.any(np.isnan(_coeffs[self.variables[i]])):
                    break
                reg_coeff[self.variables[i]] *= 10
        #todo
        if return_errors:
            predictions = {}
            errors = {}
            error_after = {}
            for i in range(self.N):
                #which one to choose? self._coeffs or _coeffs!
                #predictions[self.variables[i]] = featmat[self.variables[i]].dot(self._coeffs[self.variables[i]])
                predictions[self.variables[i]] = featmat[self.variables[i]].dot(_coeffs[self.variables[i]])
                errors[self.variables[i]] = returns[self.variables[i]] - predictions[self.variables[i]]
                error_after[self.variables[i]] = np.sum(errors[self.variables[i]]**2)/np.sum(returns[self.variables[i]]**2)
            return error_before, error_after

    def predict(self, path):
        predictions = {}        
        for i in range(self.N):
            if self._coeffs is None:
                predictions[self.variables[i]] = np.zeros(len(path["rewards"][self.variables[i]]))
            else:
                predictions[self.variables[i]] = self._features(path)[self.variables[i]].dot(self._coeffs)
        return predictions