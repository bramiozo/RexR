import pandas as pd
from collections import Counter 
import numpy as np


def get_principal_components(self, input_file=False):

	if(input_file == True):
		dat = _read_modelling_data()
	else:
		dat = self.DATA_merged
	
	# make a matrix of the data for PCA
	col_names=list(dat)
	print(col_names[24:26])
	print(dat.shape)

	df_mat = dat.iloc[:,26:]
	np_mat = df_mat.as_matrix()

	print(np_mat.shape) # I FEEL LIKE I MISS A ROW HERE
	
	print(np_mat[:5,:5])
	
	# apply PCA as on	https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
	
	#for this transformation think of the probes as oservations and the patients as dimenions (since you want to maximize the variance of genes among patients, thus transpose

	np_mat = np_mat.transpose()

	import numpy as NP
	from scipy import linalg as LA
	m, n = np_mat.shape
	# mean center the data
	np_mat -= np_mat.mean(axis=0)
	# calculate the covariance matrix
	R = NP.cov(np_mat, rowvar=False)
	print(R.shape)
	# calculate eigenvectors & eigenvalues of the covariance matrix
	# use 'eigh' rather than 'eig' since R is symmetric, 
	# the performance gain is substantial
	evals, evecs = LA.eigh(R)
	# sort eigenvalue in decreasing order
	idx = NP.argsort(evals)[::-1]
	evecs = evecs[:,idx]
	# sort eigenvectors according to same index
	evals = evals[idx]
	# in this case I take all eigenvectors because you wnat tot do a basis transformation
	evecs = evecs[:,:]
	# carry out the transformation on the data using eigenvectors
	# and return the re-scaled data, eigenvalues, and eigenvectors
	PCAtransf = NP.dot(evecs.T, np_mat.T).T
	
	# I guess now equate the common client in the three cohorts 

	# and transform back and transpose

