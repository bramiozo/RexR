import numpy as np
import pandas as pd
import scipy as sc
from scipy.spatial import pdist
from sklearn.manifold import SpectralEmbedding, MDS

from numba import jit

# original source: 
#  https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe
#  https://github.com/NikolayOskolkov/HowUMAPWorks/blob/master/HowUMAPWorks.ipynb
# edited by Bram van Es

class cumap():
	def __init__():		

		self.N_LOW_DIMS = 2
		self.MIN_DIST = 0.1
		self.N_NEIGHBOR = 15
		self.LEARNING_RATE = 1
		self.RND_SEED = 12345
		self.MAX_ITER = 200
		self.initializer = 'SE' # SE (SpectralEmbedding) or MDS
		self.N_NEIGHBOR_INIT = 50
		self.dist_fun =  'euclidean' # euclidean, minkowski, chebyshev, manhattan
		self.dist_weights = None # for Minkowski
		self.optimizer = 'CE' # CE or KL

		np.random.seed(RND_SEED)

		expr = pd.read_csv('CAFs.txt', sep='\t')
		X_train = expr.values[:,0:(expr.shape[1]-1)]
		X_train = np.log(X_train + 1)
		n = X_train.shape[0]
		print("\nThis data set contains " + str(n) + " samples")
		y_train = expr.values[:,expr.shape[1]-1]
		print("\nDimensions of the  data set: ")
		print(X_train.shape, y_train.shape)



	def get_dist(self):
		if self.dist_weights not None:
			dist = np.square(sc.spatial.pdist(X_train, metric=self.dist_fun, w=self.dist_weights))
		else:
			dist = np.square(sc.spatial.pdist(X_train, metric=self.dist_fun))

		rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]

		self.dist = dist 
		self.rho = rho

	def prob_high_dim(self, sigma, dist_row):
	    """
	    For each row of Euclidean distance matrix (dist_row) compute
	    probability in high dimensions (1D array)
	    """
	    d = self.dist[dist_row] - rho[dist_row]; d[d < 0] = 0
	    return np.exp(- d / sigma)

	def k(self, prob):
	    """
	    Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
	    """
	    return np.power(2, np.sum(prob))

	@static 
	def sigma_binary_search(k_of_sigma, fixed_k):
	    """
	    Solve equation k_of_sigma(sigma) = fixed_k 
	    with respect to sigma by the binary search algorithm
	    """
	    sigma_lower_limit = 0; sigma_upper_limit = 1000
	    for i in range(20):
	        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
	        if k_of_sigma(approx_sigma) < fixed_k:
	            sigma_lower_limit = approx_sigma
	        else:
	            sigma_upper_limit = approx_sigma
	        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
	            break
	    return approx_sigma


	def prob_low_dim(Y):
	    """
	    Compute matrix of probabilities q_ij in low-dimensional space
	    """
	    inv_distances = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
	    return inv_distances

	def CE(P, Y):
	    """
	    Compute Cross-Entropy (CE) from matrix of high-dimensional probabilities 
	    and coordinates of low-dimensional embeddings
	    """
	    Q = prob_low_dim(Y)
	    return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)

	def KL(P, Y):
	    """
	    Compute KL-divergence from matrix of high-dimensional probabilities 
	    and coordinates of low-dimensional embeddings
	    """
	    Q = prob_low_dim(Y)
	    return P * np.log(P + 0.01) - P * np.log(Q + 0.01)

	def KL_gradient(P, Y):
	    """
	    Compute gradient of KL-divergence
	    """
	    Q = prob_low_dim(Y)
	    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
	    inv_dist = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
	    return 4*np.sum(np.expand_dims(P - Q, 2) * y_diff * np.expand_dims(inv_dist, 2), axis = 1)

	def CE_gradient(P, Y):
	    """
	    Compute the gradient of Cross-Entropy (CE)
	    """
	    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
	    inv_dist = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
	    Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
	    np.fill_diagonal(Q, 0)
	    Q = Q / np.sum(Q, axis = 1, keepdims = True)
	    fact=np.expand_dims(a*P*(1e-8 + np.square(euclidean_distances(Y, Y)))**(b-1) - Q, 2)
	    return 2 * b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis = 1)


	def fit_transform(self):
		prob = np.zeros((n,n)); sigma_array = []
		for dist_row in range(n):
		    func = lambda sigma: k(prob_high_dim(sigma, dist_row))
		    binary_search_result = sigma_binary_search(func, N_NEIGHBOR)
		    prob[dist_row] = prob_high_dim(binary_search_result, dist_row)
		    sigma_array.append(binary_search_result)
		    if (dist_row + 1) % 100 == 0:
		        print("Sigma binary search finished {0} of {1} cells".format(dist_row + 1, n))
		print("\nMean sigma = " + str(np.mean(sigma_array)))

		#P = prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))
		P = (prob + np.transpose(prob)) / 2


		x = np.linspace(0, 3, 300)
		def f(x, min_dist):
		    y = []
		    for i in range(len(x)):
		        if(x[i] <= min_dist):
		            y.append(1)
		        else:
		            y.append(np.exp(- x[i] + min_dist))
		    return y

		dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))
		p , _ = optimize.curve_fit(dist_low_dim, x, f(x, self.MIN_DIST))
		a = p[0] # or pre-set a, b
		b = p[1] 

		print("Hyperparameters a = " + str(a) + " and b = " + str(b))


		if self.initializer=='SE':
			model = SpectralEmbedding(n_components = N_LOW_DIMS, n_neighbors = N_NEIGHBOR_INIT)
		else:
			model = MDS(n_components = N_LOW_DIMS)

		y = model.fit_transform(np.log(X_train + 1))
		#y = np.random.normal(loc = 0, scale = 1, size = (n, N_LOW_DIMS))

		CE_array = []
		print("Running Gradient Descent: \n")
		for i in range(MAX_ITER):
		    y = y - LEARNING_RATE * CE_gradient(P, y)
		    
		    CE_current = np.sum(CE(P, y)) / 1e+5
		    CE_array.append(CE_current)
		    if i % 10 == 0:
		        print("Cross-Entropy = " + str(CE_current) + " after " + str(i) + " iterations")

		return y