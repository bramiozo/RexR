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
* Dr. Peter Hinrich (peter.hinrich@surfnet.nl); project bios/bbmri shared datastorage/processing for diabetes

Technology:
*	NLeScienceCenter: Dr. Adriënne Mendrik
*	SURF-SARA: Dr. Peter Hinrich, can help to set-up the data infrastructure.

Facilitator:
*	Tjebbe Tauber

Business angels:
*	Helmuth van Es: multiple gentech companies
*	Fred Smulders: ingang bij Rockstart accelerator?

Partners:
*	hospitals? 
*	government?


People:
*	Tjebbe (manager)
* 	Bram (data science/q.a./privacy)
*	Sebastiaan (machine learning)
*	xxx (junior machine learning)
*	xxx (devops)
*	xxx (data viz)
*	xxx (UX)
*	xxx (developer)


# To Do
Complexity: 1, 3, 5, 7, 13
- [x] XGBOOST
- [x] DNN
- [x] CNN 
- [x] RVM
- [ ] add other decision tree methods: FACT, C4.5, QUEST, CRUISE, GUIDE
- 13, 	[ ] Combat bias corrector
- 13, 	[ ] PCA bias corrector
- 3, 	[x] simple noise addition to increase robustness (uniform distribution, single value range for entire matrix)
- 5, 	[ ] element-wise noise addition using relative value range (n percentage of absolute value)
- 3, 	[ ] n-repetitions and bagging of stochastic methods (i.e.  varying seed's)
- 1, 	[x] lightGBM
- 13, 	[ ] CEBE: Cluster-enhanced extremely biased estimator
- 13, 	[ ] PAM method (bioinformatics) http://statweb.stanford.edu/~tibs/PAM/
- 3,  	[x] generate table with classification per patient, per classification method => send to Jules
- 5, 	[ ] routine to generate table's
- 1, 	[x] top-genome selector => send to Jules
- 5, 	[ ] top-genome visualiser: top-N list -> hierarchical clustering
- 3,	[ ] patient clustering ==> all genomes, reduced
- 7,	[ ] genome clustering ==> reduced..
- ? 	[x] visualisation of training process
- 7,	[ ] user-friendly way to set-up pipelines
- 5,	[ ] add ICA for genome seperation, http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
- 7, 	[ ] add SOM for genome seperation
- 3,	[ ] sparse auto encoding for pre-processing
- 3,	[ ] t-sne / optics analyser
- 3,	[x] ROC/confusion matrix visualiser
- 5,	[x] patient similarity
- 5,	[ ] conditional survival estimator. i.e. add a regressor.
- 5,	[ ] GEO DataSets lib integration
- 7,	[ ] Make GEO datasets interactive
- 13,	[ ] refactor/optimize: Cython, numba, static def's, parallelise
- x, 	[ ] add healthy patient reference routine
- x,	[ ] healthy tissue/unhealthy tissue
- x,	[ ] add disease dependent measurement error detector/filter
- x,	[ ] cancer type detector
- x,	[ ] cancer phase detector
- x,	[ ] Image recognition 
- 5,	[ ] add genome/probeset mapping function, use docker with db
- 3, 	[ ] add plot (expression value, importance/coefficient) group by classification, labelled with genome, use Bokeh
- 3,  	[ ] add plot (number of genomes, versus accuracy)
- x,  	[ ] add graph visualisation (intra-similarity of most prominent genomes, per label)
- x,  	[ ] add measuring bias detector (multiple datasets as inputs)
- x,	[ ] add hyperoptimisation routine
- x,	[ ] add quiver visualisation for genomes
- x,	[ ] add lime visualisation for genomes
- x, 	[ ] add wrapper for (circos)[http://circos.ca/]
- x,  [ ] add option for nested cross-validation

## Datasets

* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10558
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE83744
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL97
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52581
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE11863
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31586
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66499 !
* https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80796 !

# future

* use of PyTorch, https://eng.uber.com/pyro/
* https://python-graph-gallery.com/405-dendrogram-with-heatmap-and-coloured-leaves/
* search engine for medical documents: hierarchical/DT based, human-in-the-loop
* use entity linking to fetch relevant journal papers
* build domain specific word embeddings for medical graph search
* use Siamese neural-network to get rid of the cohort bias
* add variational autoencoder? why?


# funds
* WBSO https://www.ugoo.nl/wbso-subsidie/wbso-subsidiecheck/?gclid=Cj0KCQiAzfrTBRC_ARIsAJ5ps0uImsv_6m-NiWK_jod-_XaW-8exS616zNvqDH_Pojs6MayyepqhT58aAgdiEALw_wcB
* SIDN https://www.sidnfonds.nl/aanvragen/internetprojecten

# Data protection and distribution

https://oceanprotocol.com/#why
