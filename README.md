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
* t-SNE, isomap, mds, umap
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
* successive halving, hyperband
* neural architecture searh
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
*   Dr. Harry Groen (Lung cancer)
*   Dr. Casper van Eijck (Pancreas cancer)
*	Dr. Jules Meijerink (Leukemia)
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

Partners:
* hospitals? 
* government?
* https://mytomorrows.com/nl/

Comparables/competitors:
* Precision profile: http://precisionprofiledx.squarespace.com/product-portfolio/

People:
*	(m) Tjebbe Tauber (inspiration wizard/connector)
* 	(m) Bram van Es (data science/q.a./privacy)
*	(m) Sebastiaan de Jong (machine learning)
*	(m) Evgeny (devops and machine learning)
*   (f) Nela Lekic (graph analysis/machine learning)
*	(f) Elizaveta Bakaeva (data analysis/visualisation)
*	xxx (data viz/UX)
*	xxx (bio-statistician)
*   (f) Bo (NLP)
*   PwC, DNB, AirBnB

Sources:
https://gdc.cancer.gov/
https://www.ncbi.nlm.nih.gov/
https://www.kaggle.com/c/santander-value-prediction-challenge --> high dimensionality, low sample count
https://databricks.com/product/genomics


# Done

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

# TO DO
Complexity: 1, 3, 5, 7, 13
- x,	[ ] Functionality; cancer type detector
- x,	[ ] Functionality; cancer phase detector
- x,	[ ] Functionality; Image recognition, [X-ray](https://news.mit.edu/2019/mimic-chest-x-ray-database-0201), [MRI](https://openfmri.org/dataset/)
- x,    [ ] Functionality; cancer pathway estimator
- x, 	[ ] Functionality; gene importance estimator and general factor importance tool: from weights, importance, variance explained to combinatoric importances (branch-wise importances)
- x, 	[ ] Functionality: Counter-factual explanations, (what-if scenario's)

- 5     [ ] api, GEO DataSets lib integration
- 5		[ ] api, TCGA integration
- 7     [ ] ux, Make GEO datasets interactive
- 7,	[ ] ux, user-friendly way to set-up pipelines
- x     [ ] io, add support for .vcf mutation data
- 5,	[ ] io, add genome/probeset/protein/miRNA/methyl mapping function, use docker with db (such as MonetDB, Druid or SparkSQL)
- x,	[ ] io, add containers for Neo4j
- 113,	[ ] ux/io/viz, build web interface around Superset/Druid


- 20    [ ] ml, add multi-omic combiner class: start with concatenation-based approaches
- 20    [ ] ml, add similarity class: intra and inter omic.
- 20 	[ ] ml, multi-modal learner
- 10    [ ] ml, Denoising Autoencoder
- 10	[ ] ml, Factorisation machine for imputance
- 10	[ ] ml, DeepBagNet (see Approximating CNNs with Bag-of-local-features models..)

- 30	[ ] ml, add Graph neural networks (GrapSage, DiffPool) for multi-omic analysis, [Decagon](https://cs.stanford.edu/people/jure/pubs/drugcomb-ismb18.pdf) [lit](https://cs.stanford.edu/people/jure/pubs/drugcomb-ismb18.pdf)
- 5		[ ] ml, add Generalised Additive Methods (GAM)
- 30	[ ] ml, add Neural Conditional Random Field (NCRF)
- 20	[ ] ml, add factorisation machines (FM), https://github.com/aksnzhy/xlearn
- 10	[ ] ml, Lasso, ElasticNet
- 20	[ ] ml, add Supersparse linear integer models (SLIM) https://arxiv.org/abs/1502.04269
- 10	[ ] ml,  feature augmentation:
				 - 	add transformations of the features
				 - 	add cluster-id from UMAP on raw data
				 - 	add cluster-id from graph clustering on similarity data.
				 -	add feature combinations
- 3,    [ ] ml, PCA/LDA number of components selector.
- 5,	[ ] ml, add [Generalised Additive Models](https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f) --> only works for limited number of features.
                [readme](https://multithreaded.stitchfix.com/assets/files/gam.pdf)
- 21 	[ ] ml, add support for [AutoKeras](http://autokeras.com/)
- 3,	[ ] ml, add frequent item-set analysis: association rules, A-priori, PCY (multi-stage/hash)
- 3,    [ ] ml, add factor analysis, gaussian random projection, sparse random projection
- 3,    [ ] ml, add coefficient retrieval for LDA
- 7,	[ ] ml, add hyperoptimisation routine
- 3, 	[ ] ml, FDR/MW-U loop function with noise addition to get top genomes without creating a model
- 3,    [ ] ml, add tree-based cumulative importance threshold for top genome selection
- 20,	[ ] ml. add significant factor extractor: 
			--	combine Kruskal-H with MW-U/FDR/FPR
			--  2-sided Kolmogorov-Smirnof
			--	PCA for variance explained --> sum (absolute) coefficients per feature
			--  LDA for seperation explained --> sum (absolute) coefficients per feature
			--  linear SVM/Logistic Regression: sign of importances
			-- 	tree methods for importances (use permutation importances (shap, rfpimp))

- 1,    [ ] ml, add RFECV
- 30	[ ] ml/ux, add support for [Snorkel](https://towardsdatascience.com/introducing-snorkel-27e4b0e6ecff)
- 10,	[ ] ml, add semi-supervised module (useful in case there is unlabeled data)
- 3, 	[ ] ml, element-wise noise addition using relative value range (n percentage of absolute value)
- 3,	[ ] ml, add relative noise-level
- 3,	[ ] ml, patient clustering ==> all genomes, reduced
- 7,	[ ] ml, genome clustering/community detection ==>  Sparse Affinity Propagation, Girvan-Newman Algorithm, Markov clustering, Edge Betweenness Centrality
- 10, 	[ ] ml, GAN to generate cancerous genomic profiles
- 7,	[ ] ml, UMAP / Hierarchical t-SNE / HDBSCAN / Diffusion Maps / OPTICS / Sammon mapping / LTSA , [source](https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f)
- 3,    [ ] ml, add other decision tree methods: FACT, C4.5, QUEST, CRUISE, GUIDE
- 13, 	[ ] ml, bias corrector class: COMBAT, PCA (EIGENSTRAT), DWD, L/S
- 20,	[ ] ml, patient/sample similarity/clustering based bias detection
- 13,	[ ] ml, bias detection class: between class KS/MW-U/Wasserstein/KL-divergence
- 13,   [ ] ml, outlier detector/removal: isolation forest, one-class SVM, 
- 13,   [ ] ml, add Kernel Discriminant Analysis as a non-linear feature reducer
- x,  	[ ] ml, add measuring bias detector (multiple datasets as inputs)
- 13, 	[ ] ml, CEBE: Cluster-enhanced extremely biased estimator
- 13, 	[ ] ml, PAM method (bioinformatics) http://statweb.stanford.edu/~tibs/PAM/
- 5,	[ ] ml, add ICA for genome seperation, http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
				with ICA you can find commonalities in different groups.

- 20,	[ ] ml, ICA/CCA module for multi-omic feature analysis
- 10,	[ ] ml, polynomial expansion module for multi-omic feature combinations
- 7, 	[ ] ml, add SOM for genome seperation
- 10,   [ ] ml, add Occam factor function to extract approximation of model complexity
- 3,	[ ] ml, multilayer sparse auto encoding for pre-processing and feature detection, and DAE for denoising
- x,    [ ] ml, add iCluster(?), in [R](https://cran.r-project.org/web/packages/iCluster/iCluster.pdf)
- 5,	[ ] ml, conditional survival estimator. i.e. add a regressor, Kaplan-Meier
- 13,	[ ] ml, refactor/optimize: Cython, numba, static def's, parallelise, modularize
- x, 	[ ] ml, add healthy patient reference routine
- x,	[ ] ml, healthy tissue/unhealthy tissue
- x,	[ ] ml, add disease dependent measurement error detector/filter
- x,  	[ ] ml, add option for nested cross-validation
- x, 	[ ] ml, add a posteriori accuracy checker
- x,	[ ] ml, add Automatic Relevance Determination (ARD), Bayesian Discriminative Modelling.
- x, 	[ ] ml, add support for image based classification: test on kaggle set, https://www.kaggle.com/c/data-science-bowl-2018/data
- x,	[ ] ml, add support for time series based classification: test on EEG kaggle set, https://www.kaggle.com/c/grasp-and-lift-eeg-detection MyFly (CNN, LSTM): add TCN, GRU support
- x,    [ ] ml, add "deep dreaming": or sample generator functionality given a classification label generate a representative sample.		
- 15,   [ ] ml, Add graph abstraction: [source](https://github.com/theislab/graph_abstraction), [source](https://scanpy.readthedocs.io/en/latest/)
            , MST (Kruskal)

****
- 3,	[ ] viz, add missing data visualizer, https://github.com/ResidentMario/missingno
- 3, 	[ ] viz, add tree visualiser, https://github.com/parrt/dtreeviz
- 5,	[ ] viz, add parallel coordinates to visualise 'pathways':  inflate height on dim axes by taking Hadamard power.
- ? 	[ ] viz, visualisation of training process
- 3, 	[ ] viz, add plot (expression value, importance/coefficient) group by classification, labelled with genome, use Bokeh
- 3,  	[ ] viz, add plot (number of genomes, versus accuracy)
- x,  	[ ] viz, add graph visualisation (intra-similarity of most prominent genomes, per label)
- x,	[ ] viz, add quiver visualisation for genomes, also see https://distill.pub/2018/building-blocks/
- x,	[ ] viz, add LIME/DeepLift visualisation for model explanations of neural net's (https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
- x, 	[ ] viz, add SHAP visualisation for the model explanation of tree methods, 
- x, 	[ ] viz, add visualisation of cumulative importance of tree branches.
- x, 	[ ] viz, add tree interpreter (permutation importances) ELI5 https://github.com/TeamHG-Memex/eli5
- x,	[ ] viz, add model interpreter (shapely values) SHAP https://github.com/slundberg/shap
- x, 	[ ] viz, add Additive feature attribution methods
- x,    [ ] viz, model explainability: using L2X, QII and additive index models (xNN)
- x,	[ ] viz, train simple model on complex model (GBT-> single DT regressor on proba's)
- x, 	[ ] viz, partial model dependence plots for clinical data and individual conditional expectation
- x, 	[ ] viz, add correlation graphs: corr --> networktools
- x,	[ ] viz: https://www.kaggle.com/kanncaa1/rare-visualization-tools
- x, 	[ ] viz: https://www.kaggle.com/mirichoi0218/classification-breast-cancer-or-not-with-15-ml
- 5, 	[ ] viz, routine to generate heatmap table's
- 20,	[ ] viz, Treat as 2D classification problem, and visualize with Quiver, get inspiration from this [playground](https://playground.tensorflow.org/)
- 5, 	[ ] viz, top-genome visualiser: top-N list -> hierarchical (agglomerative) clustering
- 20,	[ ] viz, Treat as 2D classification problem, and visualize with Quiver.
- 5, 	[ ] viz, top-genome visualiser: top-N list -> hierarchical (agglomerative) clustering [seaborne](https://seaborn.pydata.org/examples/structured_heatmap.html)
- 10,	[ ] viz, genome/patient clustering using [Vega](https://vega.github.io/vega/examples/edge-bundling/), [Altair](https://altair-viz.github.io/) or [D3js](https://beta.observablehq.com/@mbostock/d3-hierarchical-edge-bundling)
- x, 	[ ] viz, add wrapper for (circos)[http://circos.ca/]
- x,	[ ] viz, add lgbm/xgb/rf model visualisation
- x,	[ ] viz, Datawrapper, LocalFocus, Flourish


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

* https://python-graph-gallery.com/405-dendrogram-with-heatmap-and-coloured-leaves/
* search engine for medical documents: hierarchical/DT based, human-in-the-loop
* use entity linking to fetch relevant journal papers
* build domain specific word embeddings for medical graph search
* use Siamese neural-network to get rid of the cohort bias
* use [Kubeflow](https://hackernoon.com/what-do-you-call-ai-without-the-boring-bits-8861760bf5e) for pipelining
* add meta classifier: UMAP embedding+Convex hull+MSP+SVM, 
* add meta classifier (see [Matching Nets](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf),
and [Relation Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)
and [Prototypical Networks](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)): SAE or UMAP embedding+class-matching (Rank correlation, similarity, Wasserstein distance or softmax of distance) with Barycentered sample. Also see [this](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#optimization-based) overview.
* add MAML/Reptile to speed up learning
* add image-caption generator to evaluate images?
* add functionality for the practitioner to draw a decision plane to manually create a predictor
* add visualisation of phenotypical manifolds in omics-space and position of patient in that space.
* Neuro-conditional random field for tumor detection(research.baidu.com/Blog/index-view?id=104)
* contact https://turbine.ai/: they can simulate the effect of anti-tumour medication

# funds
* WBSO https://www.ugoo.nl/wbso-subsidie/wbso-subsidiecheck/?gclid=Cj0KCQiAzfrTBRC_ARIsAJ5ps0uImsv_6m-NiWK_jod-_XaW-8exS616zNvqDH_Pojs6MayyepqhT58aAgdiEALw_wcB
* SIDN https://www.sidnfonds.nl/aanvragen/internetprojecten
* KPN/Menzis/Monuta: https://fd.nl/economie-politiek/1239055/nieuw-fonds-met-durfkapitaal-voor-zorgstart-ups
* Blue Sparrows MedTech Fonds
* eScience https://www.esciencecenter.nl/funding/big-data-health

# Data protection and distribution

https://oceanprotocol.com/#why
