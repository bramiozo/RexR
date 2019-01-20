# Cardio hackaton

Main goal and background: 
to diagnose a cardial problem there are multiple diagnostic tools;
* physiological: blood testing, red blood cell, pulsation, oedeem etc.
* ECG (electrocardiogram), starting in the ambulance with three electrodes to the hospital with twelve electrodes
* CT (computer tomography), 3D scan of the heart using X-rays: expensive and increases risk of cancer, limited availability
* NMRI or SPECT imaging: limited availability

It can occur that a patient goes through all these diagnostics and turns out be clear of cardial problems, according to the cardiologist.
We want to detect these non-cardial problems based on all the available data *prior* to the CT and the NMRI/SPECT imaging.

Administrative: Every team member participating in the hackathon has to 
* sign a non-disclosure agreement
* participate in a UMCU introduction session of three hours: there is one every first work day of the month

Request from every participant:
* indicate availability for December 3 and willingness/ability to join the introduction session, the next on will be in January
* indicate availability between December 3 and December 12 to mine the radiology/cardiology reports, I expect 1/2 days of work (haha)
* read up on the use of PySpark
* send me a list of Python libraries we think we need for the hackathon as we don't have root access they will need to install it

Architecturally:
* Hadoop stack: MapR implementation
* Python with Jupyter notebooks
* Distributed computing using PySpark
..links to tutorials..

Data sets:
* clinical: diagnostic history; past issues and treatments
* clinical: medication table; detailed list of medication, including ontology
* measurement: ECG; pre-processed quasi-timeseries data
* measurement: cell dynamics; the output of a cell measurement device that uses lasers and spectography
* measurement: blood measurements; blood values for minerals/white & red bloodcells, etc.
* descriptive: radiology reports; concise report of radiologists describing outcome of CT/SPECT or NMRI.
* descriptive: cardiology reports; summarising report of cardiologist, including a so-called anamnese, 
conclusion stating whether or not the patient has cardial or a non-cardial problems.

Deliverables:
Challenge 1: extracting features **and** labels from the descriptive data (the radiology/cardiology reports)
Challenge 2: cleaning the non-measurement data
Challenge 3: create predictors based on the phenotypical data and the measurement data. 
The primary target variable is: cardial/non-cardial.
Challenge 4: get useful insights from the models 
Synthesis: takeaways lessons with regard to data storage/presentation/architecture and modelling

Subgoals:
* find biomarkers from the cell dynamics and lab measurement data: interact with Imo on this
* verify hypothesis that 


Basic literature:
* (ECG)[https://www.theheartcheck.com/documents/ECG%20Interpretation%20Made%20Incredibly%20Easy!%20(5th%20edition).pdf]
* (CT 1)[https://www.lf2.cuni.cz/files/page/files/2014/basic_principles_of_ct.pdf]
* (the heart)[https://www.imaios.com/en/e-Anatomy/Thorax-Abdomen-Pelvis/Heart-pictures]


# Plot and caveats, plan B to Z

In the feature mining process we will be biased towards the low-hanging fruit, i.e. the statistically most 
salient and prominent features. Due to time constraints and due to the inherent limitations of 
NLP in lieu of broken text and due to the fact that there is no large Dutch medical corpus available.

Assuming we are succesfull in the feature mining process we construct an interpretable multi-modal model.

Figures of merit:
Sensitivity
Specifity
F1 score
Positive-predictive-value

The model results should be made insightful using tools like LIME/DeepLift/SHAP/ELI5/rfpimp.

Initially
Faust and Raffaele are in team data prep and analytics
Sebastiaan en Evgeny are in team pipeline
Eliza, Tjebbe en Merel are in team business
Bram will be mostly working extracting features from the treatment/medical data 

*Intermediate solution*, if the resulting prediction is uncertain: There are apps/devices available for single-direction ECG measurements
The data of this device can then be used to improve the accuracy of the predictor. Also, using only single-lead ECG measurements an above-human 
accuracy level has been attained for the classification of the rythm: https://arxiv.org/pdf/1707.01836.pdf, https://www.nature.com/articles/s41591-018-0268-3
Future work: For the interpretability of the classification we can give the probability for a particular type of ECG.
In general we can use supervised learning on the seperate datasets to classify the data and provide context for the final classification.

- Symptoms: Difficult to extract net polarity of symptoms from short anamneses written in various styles
- Ontologie: we have the ATC code for medication, we can use this reduce the medical history
- Treatments: if too sparse, ignore for ML model, keep for phenotypical EDA
- Ergometrie: estimate extraction of hart rate/wattage under load
- Riskfactors: estimate extraction of riskfactors 

The data quality can be much improved if cardiologists and radiologists respect an ontology and avoid ambiguity, for instance through structured anamnese/ergometric/riskfactor forms with (hierarchical dropdowns). Especially hard metrics like the hart rate versus wattage or discrete classifications/observation like ST depression are QRS widening etc.

In the current setup we had to infer/guess rules for text mining which create an unnecessary source of error for the model creation.


## Spark MLlib

(pyspark.ml reference)[https://spark.apache.org/docs/2.1.1/api/python/pyspark.ml.html#module-pyspark.ml]

We will make (pipelines)[https://spark.apache.org/docs/2.1.3/ml-pipeline.html] like 

```
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)
```

for us:
1. data normalisation
2. outlier removal
3. dim reduction
4. polynomial expansion
5. model selection
6. hyperoptimisation

Spark MLlib also has a cross validator and a grid search option, check out (the pyspark reference)[https://spark.apache.org/docs/2.1.3/ml-tuning.html].

Most relevant for us are:

* pyspark.ml.feature.Binarizer
* pyspark.ml.feature.Bucketizer
* pyspark.ml.feature.OneHotEncoder
* pyspark.ml.feature.PCA
* pyspark.ml.feature.PolynomialExpansion
* pyspark.ml.classification.GBTClassifier
* pyspark.ml.classification.RandomForestClassifier


# Spark SQL and dataframes


# Considerations

- instead of manually extraction text features from reports, rather vectorize and embed the 
text and apply a supervised ML model on the text 
- perform full feature engineering + modelling in one pipeline