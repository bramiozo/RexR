# Cardio hackaton

Main goal and background: 
to diagnose a cardial problem there are multiple diagnostic tools;
* physiological: blood testing, red blood cell, pulsation, oedeem etc.
* ECG (electrocardiogram), starting in the ambulance with three electrodes to the hospital with twelve electrodes
* CT (computer tomography), 3D scan of the heart using X-rays: expensive and increases risk of cancer, limited availability
* NMRI or SPECT imaging, super expensive, very limited availability

It can occur that a patient goes through all these diagnostics and turns out be clear of cardial problems, according to the cardiologist.
We want to detect these non-cardial problems based on all the available data *prior* to the CT and the NMRI/SPECT imaging, 
and prior to NMRI/SPECT.


Administrative: Every team member participating in the hackathon has to 
* sign a non-disclosure agreement
* participate in a UMCU introduction session of three hours: there is one every first work day of the month

Architecturally:
* Hadoop stack: MapR implementation
* Python with Jupyter notebooks
* Distributed computing using PySpark

Data sets:
* phenotypical: diagnostic history; past issues and treatments
* phenotypical: medication table; detailed list of medication, including ontology
* measurement: ECG; pre-processed quasi-timeseries data
* measurement: cell dynamics; the output of a cell measurement device that uses lasers and spectography (unknown what the values mean, exactly)
* measurement: blood measurements; blood values for minerals/white & red bloodcells, etc.
* descriptive: radiology reports; concise report of radiologists describing outcome of CT/SPECT or NMRI.
* descriptive: cardiology reports; summarising report of cardiologist, including a so-called anamnese, 
conclusion stating whether or not the patient has cardial or a non-cardial problems.

Deliverables:
Challenge 1: extracting features **and** labels from the descriptive data (the radiology/cardiology reports)
Challenge 2: cleaning the non-measurement data
Challenge 3: create predictors based on the phenotypical data and the measurement data. 
The primary target variable is: cardial/non-cardial.
Challenge 4: create interface for these predictors, plus create visual output.


Request from every participant:
* indicate availability for December 3 and willingness/ability to join the introduction session, the next on will be in January
* indicate availability between December 3 and December 12 to mine the radiology/cardiology reports, I expect 1/2 days of work
* read up on the use of PySpark
* send me a list of Python libraries we think we need for the hackathon as we don't have root access they will need to install it


Basic literature:
* (ECG)[https://www.theheartcheck.com/documents/ECG%20Interpretation%20Made%20Incredibly%20Easy!%20(5th%20edition).pdf]
* (CT 1)[https://www.lf2.cuni.cz/files/page/files/2014/basic_principles_of_ct.pdf]
* (the heart)[https://www.imaios.com/en/e-Anatomy/Thorax-Abdomen-Pelvis/Heart-pictures]

