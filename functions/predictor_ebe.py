# Cluster Enhanced Convex Biased Estimator
# binomial classification

class CEBE():
	def __init__(self, clustering = "SOM", dim_reduction = None, aggregations = None, classifier = "LSVM"):

# Clustering
####################################
# SOM, AP




# Dim. reduction
####################################
# Variance filter per cluster?



# Sample aggregations per cluster
####################################
# percentiles: 10, 25, 50, 75, 90, modes: peaks in density estimations
# percentiles per class, per feature, per cluster,
# [cluster_n, classification_m, percentiles, ..]



# Convex Hulls
####################################
# 
# [hull_k, cluster_n, classification_m, hull_coordinates_mean, hull_coordinates_median, hull_coordinates_variance]





