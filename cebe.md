# CEBE

=== Algorithm ===
# Randomize the  node weight vectors in a map
# Randomly pick an input vector <math>{D}(t)</math>
# Traverse each node in the map
## Use the [[Euclidean distance]] formula to find the similarity between the input vector and the map's node's weight vector
## Track the node that produces the smallest distance (this node is the best matching unit, BMU)
# Update the weight vectors of the nodes in the neighborhood of the BMU (including the BMU itself) by pulling them closer to the input vector
## <math>W_{v}(s + 1) = W_{v}(s) + \theta(u, v, s) \cdot \alpha(s) \cdot (D(t) - W_{v}(s))</math>
# Increase <math>s</math> and repeat from step 2 while <math>s < \lambda</math>

A variant algorithm:
# Randomize the map's nodes' weight vectors
# Traverse each input vector in the input data set
## Traverse each node in the map
### Use the [[Euclidean distance]] formula to find the similarity between the input vector and the map's node's weight vector
### Track the node that produces the smallest distance (this node is the best matching unit, BMU)
## Update the nodes in the neighborhood of the BMU (including the BMU itself) by pulling them closer to the input vector
### <math>W_{v}(s + 1) = W_{v}(s) + \theta(u, v, s) \cdot \alpha(s) \cdot (D(t) - W_{v}(s))</math>
# Increase <math>s</math> and repeat from step 2 while <math>s < \lambda</math>

# sources
* https://arxiv.org/pdf/0709.3461.pdf
* http://peterwittek.com/somoclu-in-python.html


Quick pairwise distance estimator:
use artificial reference vector filled with random values.
first determine feature ranges, use as bounds for random selections from uniform distributions

1. noise cluster, small distance to reference vector, take out?
2. after removal, determine distribution of distance from ref. vector
3. check modes. if larger than 1 apply SOM
