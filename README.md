# RexR
Repos for RexRocket, and how to apply ML to high-dimensional problems with a small sample count

Aspired functions contained in the library are

Pre/post-processing
*	Data coupling --> with genetic databases
*	Data imputance
*	Probeset cleaning
*	Cohort/lab correction
*	Gene prioritisation (top N genomes)
*	Patient clustering


Analysis methods
* PCA, LDA, PLS, QDA, Autoencoding
* self-organising maps
* Hierarchical clustering
* t-SNE, isomap, mds
* affinity propagation

Prediction
* ensemble learning
* multi-layer neural network
* deep learning
* tree-based algorithms: extraTrees, random forest, C5.0, CART, XGB, LightGBM
* novel Cluster-enhanced extremely-biased estimator (CEBE)

hyperlearning
* simulated annealing
* genetic algorithm
* Bayesian optimisation
* grid search (1)
* random selection (3)
* successive halving
* active learning (2) --> output difficult classes and output test samples that
                          need labeling

Visualisation
* 	gene importance using graphs


# Possible upgrades

* addition of image analysis/classification and the combination with genomic expression profiles

# Possible techniques
* increase robustness: Apply data augmentation such affine transformations, after mapping genome vector to surface
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* apply biased estimator using aggregations of genomic vectors

# Possible collaborations

Science:
*	Jules Meijerink; principal investigator
*	Dr. Mohammed El Kebir; computational biologist 
*	Dr. Gunnar W. Klau; computational biologist
*	Dr. Marc Deisenroth; trust and transparancy in ML
* 	Dr. Peter Hinrich (peter.hinrich@surfnet.nl); project bios/bbmri shared datastorage/processing for diabetes

Technology:
*	NLeScienceCenter: Dr. Adriënne Mendrik
*	SURF-SARA: Dr. Peter Hinrich, can help to set-up the data infrastructure.

Facilitator:
*	Tjebbe Tauber

Business angels:
*	Helmuth van Es: multiple gentech companies
*	Fred Smulders: ingang bij Rockstart accelerator?


Piggyback:
*	Deloitte


# To Do
- [x] XGBOOST
- [x] DNN
- [x] CNN 
- [x] RVM
- [ ] lightGBM
- [ ] CEBE: Cluster-enhanced extremely biased estimator
- [ ] PAM method
- [x] generate table with classification per patient, per classification method => send to Jules
- [ ] routine to generate table's
- [ ] top-genome selector => send to Jules
- [ ] top-genome visualiser: top-N list -> hierarchical clustering
- [ ] patient clustering ==> all genomes, reduced
- [ ] genome clustering ==> reduced..
- [ ] visualisation of training process
- [ ] user-friendly way to set-up pipelines
- [ ] add ICA for genome seperation, http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
- [ ] add SOM for genome seperation
- [ ] sparse auto encoding
- [ ] t-sne / optics analyser
- [x] ROC/confusion matrix visualiser
- [x] patient similarity
- [ ] cohort-bias reducer
- [ ] conditional survival estimator
- [ ] GEO DataSets lib integration
- [ ] gene importance visualiser
- [ ] refactor/optimize: Cython, numba, static def's, parallelise