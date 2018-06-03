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
* affinity propagation, community detection
* cancer similarity based on open data

Prediction
* ensemble learning
* deep learning, both for classification and regression.
* simple (but descriptive) methods: GPC, lSVM, LR etc.
* tree-based algorithms: extraTrees, random forest, C5.0, CART, XGB, LightGBM
* novel Cluster-enhanced extremely-biased estimator (CEBE)

Hyperlearning
* simulated annealing
* genetic algorithm
* Bayesian optimisation
* grid search 
* random selection 
* successive halving
* active learning -> output difficult classes and output test samples that
                          need labeling (interactive)

Visualisation
* 	gene importance using graphs
*   gene cluster identification
*   patient cluster identification


# Possible upgrades

* addition of image analysis/classification and the combination with genomic expression profiles

# Possible techniques
* increase robustness: Apply data augmentation such affine transformations, after mapping genome vector to surface
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* apply biased estimator using aggregations of genomic vectors

# Possible collaborations

Science:
*	Dr. Jules Meijerink; principal investigator
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
*	Tjebbe Tauber (manager)
* 	Bram van Es (data science/q.a./privacy)
*	Sebastiaan de Jong (machine learning)
* 	Kia Eisinga (data science)
*	Evgeny? (devops)
*	Roberto Moratore 
*	xxx (data viz/UX)
*	xxx (bio-statistician)

Sources:
https://gdc.cancer.gov/
https://www.ncbi.nlm.nih.gov/



# To Do
Complexity: 1, 3, 5, 7, 13
- [x] XGBOOST
- [x] DNN
- [x] CNN 
- [x] RVM
- [x] simple noise addition to increase robustness (uniform distribution, single value range for entire matrix)
- [x] lightGBM
- [x] generate table with classification per patient, per classification method => send to Jules
- [x] top-genome selector => send to Jules
- [x] ROC/confusion matrix visualiser
- [x] patient similarity
- [x] add false positive rate (sklearn.feature_selection.SelectFpr)
- [x] n-repetitions and bagging of stochastic methods (i.e.  varying seed's)


- 5     [ ] GEO DataSets lib integration
- 7     [ ] Make GEO datasets interactive
- 7,	[ ] add hyperoptimisation routine
- 3, 	[ ] FDR/MW-U loop function with noise addition to get top genomes without creating a model
- 1,    [ ] add RFECV
- 3, 	[ ] element-wise noise addition using relative value range (n percentage of absolute value)
- 3,	[ ] add relative noise-level
- 5, 	[ ] top-genome visualiser: top-N list -> hierarchical clustering
- 3,	[ ] patient clustering ==> all genomes, reduced
- 7,	[ ] genome clustering ==> reduced..
- 7,	[ ] user-friendly way to set-up pipelines
- 10, 	[ ] GAN to generate cancerous genomic profiles
- 7,	[ ] Treat as 2D classification problem, and visualize with Quiver.


- 3,    [ ] add other decision tree methods: FACT, C4.5, QUEST, CRUISE, GUIDE
- 13, 	[ ] Combat bias corrector
- 13, 	[ ] PCA bias correcto
- 13, 	[ ] CEBE: Cluster-enhanced extremely biased estimator
- 13, 	[ ] PAM method (bioinformatics) http://statweb.stanford.edu/~tibs/PAM/
- 5, 	[ ] routine to generate table's
- ? 	[ ] visualisation of training process
- 5,	[ ] add ICA for genome seperation, http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
- 7, 	[ ] add SOM for genome seperation
- 3,	[ ] sparse auto encoding for pre-processing and feature detection
- 3,	[ ] t-sne / optics analyser
- 5,	[ ] conditional survival estimator. i.e. add a regressor.
- 13,	[ ] refactor/optimize: Cython, numba, static def's, parallelise, modularize
- x, 	[ ] add healthy patient reference routine
- x,	[ ] healthy tissue/unhealthy tissue
- x,	[ ] add disease dependent measurement error detector/filter
- x,	[ ] cancer type detector
- x,	[ ] cancer phase detector
- x,	[ ] Image recognition
- 5,	[ ] add genome/probeset mapping function, use docker with db (such as MonetDB, Druid or SparkSQL)
- 3, 	[ ] add plot (expression value, importance/coefficient) group by classification, labelled with genome, use Bokeh
- 3,  	[ ] add plot (number of genomes, versus accuracy)
- x,  	[ ] add graph visualisation (intra-similarity of most prominent genomes, per label)
- x,  	[ ] add measuring bias detector (multiple datasets as inputs)
- x,	[ ] add quiver visualisation for genomes, also see https://distill.pub/2018/building-blocks/
- x,	[ ] add lime visualisation for genomes
- x, 	[ ] add wrapper for (circos)[http://circos.ca/]
- x,  	[ ] add option for nested cross-validation
- x,	[ ] add containers for Neo4j
- x, 	[ ] add a posteriori accuracy checker
- x,	[ ] add Automatic Relevance Determination (ARD), Bayesian Discriminative Modelling.
- x,	[ ] add lgbm/xgb/rf model visualisation
- x,	[ ] viz: https://www.kaggle.com/kanncaa1/rare-visualization-tools
- x, 	[ ] viz: https://www.kaggle.com/mirichoi0218/classification-breast-cancer-or-not-with-15-ml
- x, 	[ ] add support for image based classification: test on kaggle set, https://www.kaggle.com/c/data-science-bowl-2018/data
- x,	[ ] add support for time series based classification: test on EEG kaggle set, https://www.kaggle.com/c/grasp-and-lift-eeg-detection
			MyFly (CNN, LSTM): add TCN, GRU support
- x,    [ ] add "deep dreaming": or sample generator functionality given a classification label generate a representative
            sample.


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
* add variational autoencoder?


# funds
* WBSO https://www.ugoo.nl/wbso-subsidie/wbso-subsidiecheck/?gclid=Cj0KCQiAzfrTBRC_ARIsAJ5ps0uImsv_6m-NiWK_jod-_XaW-8exS616zNvqDH_Pojs6MayyepqhT58aAgdiEALw_wcB
* SIDN https://www.sidnfonds.nl/aanvragen/internetprojecten

# Data protection and distribution

https://oceanprotocol.com/#why
