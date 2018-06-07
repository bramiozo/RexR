
'''
Omdat er meerdere metingen per patient zijn, maken ze van gegevens eerst een soort mini tijdseries. Daarna gebruiken ze een analyse die random forest heet om te kijken welke factoren bepalen of een patiënt over tijd een afstoting krijgt. Dit zelfde doen ze om te zoeken naar verschillen tussen de factoren die afstoting voorspellen bij patienten met de duale therapie vs patienten die enkele therapie krijgen. Uit deze analyse komen altijd wel voorspellende factoren, de vraag is echter of het ook krachtige voorspellers zijn.

3 onderzoeksvragen:

1.        Kun je voorspellen welke patient een afstoting krijgt

2.        Kun je voorspellen welke patient afstoting krijgt na stoppen duale therapie

3.        Is leeftijd een voorspellende factor?

Het voordeel van deze aanpak, is dat je na een dag weet of het aannemelijk is dat er een antwoord komt op deze vragen, met de huidige data set. Het nadeel is dat je pas echt weet hoe sterk de voorspellende waarde is als je gaat modeleren of een neuraal netwerk maakt. Dat kost meer tijd. Ook zit hier nog niet het analyseren van de ruwe data in en daar zijn we toch ook wel nieuwsgierig naar.

Dit lijkt me een goede eerste stap. Het vraagt nog weinig tijdsinspanning van ons allen en laat wel zien wat er mogelijk is. Na deze dag kunnen Bram en Sebastiaan beter een inschatting maken hoe de andere vragen te benaderen. En of dat kansrijk is. Jullie vertelde dat de huidige voorspellers zo rond de 60% accuraat waren, daar moeten we wel boven kunnen komen natuurlijk.

Wat vinden jullie?

Als jullie dit een goed plan vinden, zijn de praktische stappen:

1.        Afstemmen met data compliance officer

a.        weet niet hoe deze functie bij jullie heet, maar zal neerkomen op contractje over goed met data omgaan en dat er geen uurloon wordt betaalt

2.        Datum prikken

a.        eerste uur vd dag beetje hulp van jullie bij instaleren, koffie apparaat vinden en wat toelichting op de data. Daarna gaan ze zelf aan de slag. Het is wel makkelijk als ze iemand hebben om medische vragen aan te stellen gedurende de dag.

3.        Evalueren
'''

# Approach

We are dealing with a timeseries classification problem that can be approached with various complexity.
Firstly we can use the aggregated data per time unit which produced a severely reduce dataset out of the box.
Furthermore, we can remove the importance of the time ordering by creating aggregate time features such as frequency, 
entropy, mean, variance, discrete wavelet transform. This enables the use of standard non-timeseries specific methods such as Random Forest models (, LGBM, XGB, CatBoost or ET).

Secondly, we can apply simple timeseries classifier on the aggregated ordered data

Thirdly and finally, we can apply a timeseries classifier on the original images:
* by treating each pixel location as a channel: perhaps we need to reduce all pixel location to normalised locations, using
affinity propagation, k-means, etc.
* through unsupervised learning we can find clusters per image type, each time unit has one cluster label per image type.
* similar to the first approach, but instead we apply aggregation methods on the raw images (DTW, DWT, spectral analysis, etc.)