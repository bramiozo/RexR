import pandas as pd
from collections import Counter 
import numpy as np
from sklearn import discriminant_analysis, decomposition

def get_principal_components(X, y, n_comp):
	X, y = decomposition.PCA(n_components=n_comp, 
							copy=True, whiten=False, 
							svd_solver='auto', 
							tol=0.0, iterated_power='auto', 
							random_state=None).fit_transform(X, y)

	return X,y

def get_linear_discriminant_analysis(X, y):


	return lda_transformed

def get_quadrant_discriminant_analysis(X, y):

	return qda_transformed

def get_vector_characteristics():
	
	return True

def get_genome_variation():

	return True


