# Hackathon 2018

# Background..


# Data sets

We have two collections of data sets, one for melanoma tumors and one for lung cancer 
tumors. We will focus our efforts on the melanoma tumors. Although we will 
produce melanoma-specific results, the underlying methods should be equally applicable to the 
lung cancer data set as the data schema's are identical.

For both the melanoma and the lung cancer data set we have:
* DNA, mutations: genomics
* DNA, Copy number variations: genomics
* DNA, Methylation: epigenomics
* RNA, genomic expression: epigenomics
* RNA, miRNA: transcriptomics
* Proteins: proteomics

## DNA, [Mutation](https://ghr.nlm.nih.gov/primer/mutationsanddisorders/possiblemutations)

Literally, per genome and chromosome the change in the pair compared 
to a normal reference. Remember we have (Adenine,Thymine) and (Guanine,Cytosine) as the base pairs.

The types of mutations include (taken [from here]((https://ghr.nlm.nih.gov/primer/mutationsanddisorders/possiblemutations)):

**Missense mutation**: This type of mutation is a change in one DNA base pair that results in the substitution of one amino acid for another in the protein made by a gene. 

**Nonsense mutation**: is also a change in one DNA base pair. Instead of substituting one amino acid for another, however, the altered DNA sequence prematurely signals the cell to stop building a protein. This type of mutation results in a shortened protein that may function improperly or not at all.

**Insertion**: An insertion changes the number of DNA bases in a gene by adding a piece of DNA. As a result, the protein made by the gene may not function properly.

**Deletion**: A deletion changes the number of DNA bases by removing a piece of DNA. Small deletions may remove one or a few base pairs within a gene, while larger deletions can remove an entire gene or several neighboring genes. The deleted DNA may alter the function of the resulting protein(s).

**Duplication**: A duplication consists of a piece of DNA that is abnormally copied one or more times. This type of mutation may alter the function of the resulting protein.

**Frameshift mutation**: This type of mutation occurs when the addition or loss of DNA bases changes a gene's reading frame. A reading frame consists of groups of 3 bases that each code for one amino acid. A frameshift mutation shifts the grouping of these bases and changes the code for amino acids. The resulting protein is usually nonfunctional. Insertions, deletions, and duplications can all be frameshift mutations.

**Repeat expansion**: Nucleotide repeats are short DNA sequences that are repeated a number of times in a row. For example, a trinucleotide repeat is made up of 3-base-pair sequences, and a tetranucleotide repeat is made up of 4-base-pair sequences. A repeat expansion is a mutation that increases the number of times that the short DNA sequence is repeated. This type of mutation can cause the resulting protein to function improperly.

### DATA FIELDS, shape (422553, 11)
```ID      |  Location        | Change     |  Gene   | Mutation type|  Var.Allele.Frequency  | Amino acid```

```SampleID,| Chr, Start, Stop|  Ref, Alt  | Gene    |    Effect    |  DNA_VAF, RNA_VAF      | Amino_Acid_Change```

```string   |string, int, int | char, char | string  |    string    |  float, float          |  string```

NOTE: this gives us direct insight in how genetic mutations lead to changes in amino-acids.

## Copy Number Variations

A copy number variation (CNV) is when the number of copies of a particular gene varies from one individual to the next.

### DATA FIELDS, shape (24802, 372)
```Gene      | Chr, Start, Stop | Strand     |   SampleID 1..SampleID N```

```string    |string, int, int  | int        |  int..int```


## Methylation, gene expression regulation

Degree of [methylation](https://en.wikipedia.org/wiki/DNA_methylation)
indicates addition of Methyl groups to the DNA. Increased methylation is associated with less transcription of the DNA:
Methylated means the gene is switched OFF, Unmethylated means the gene is switched ON.

Alterations of DNA methylation have been recognized as an important component of cancer development.


### DATA FIELDS, shape (485577, 483) 
```probeID   | Chr, Start, Stop | Strand  | Gene   |  Relation_CpG_island | SampleID 1..SampleID N```

```string    |string, int, int  | int     | string |   string             | float..float```


## RNA, gene expression

Again four building blocks; Adenosine (A), Uracil (U), Guanine (G), Cytosine (C).

(DNA) --> (RNA)

A --> U 

T --> A

C --> G

G --> C

Gene expression profiles, continuous values resulting from the normalisation of counts.

### DATA FIELDS, shape (60531, 477)
```Gene      | Chr, Start, Stop | Strand  | SampleID 1..SampleID N```

```string    |string, int, int  | int     |  float..float```


## miRNA, transcriptomics

The connection between the RNA production and protein creation. I.e. perhaps miRNA expression values can be associated with specific proteins.

### DATA FIELDS, shape (2220, 458)
```MIMATID  | Name   | Chr, Start, Stop | Strand  | SampleID 1..SampleID N```

```string   | string |string, int, int  | int     |  float..float```


## Proteomes

Proteine expression profiles, ditto, continuous values resulting from the normalisation of counts


### DATA FIELDS, shape (282, 355)
```ProteinID  | SampleID 1..SampleID N```

```string     | float..float```

### QUIZ, identify our data sets in the following image!


![image.png](_images/overview.png)


## GOAL

Some degree of multi-omic analysis and identification of pathways.

![image.png](_images/multi_omic.png)

# Targets

First we need get a picture of what a "signature" actually means in this context. We basically have hierarchically dependent data with "pathways" going through those layers, those pathways are connected by mutations on the one end (DNA) and proteins on the other end. How to find those pathways is the main question, because once we can do that, we only have to identify either which pathways are typical for people that do or do not respond well to immunotherapy, or what part of the pathway is typically different for those patients.

So what is a pathway? A pathway is a chain of molecular changes that leads to, in our case, the production of certain proteines, 
or (since we don't have many proteomic measurements) certain RNA codes. In the simplest form it is a sequence, but it 
is more likely similar to a bi-directed graph, in it's simplest form; DNA mutation <--> RNA <--> mRNA <--> proteines.
The collection of molecular regulators that govern the genomic expression levels of mRNA and proteins is called the 
[gene regulatory network (GRN)](https://en.wikipedia.org/wiki/Gene_regulatory_network). So, instead of chain, it is better
to say a network of molecular changes.

![GRN](_images/Gene_Regulatory_Network_2.jpg)

![GRN](_images/Gene_Regulatory_Network.jpg)

![GRN](_images/DG_Network_in_Hybrid_Rice.png)

Specifically, given that we find a pathway, the genomes that, given a mutation, will lead to a proto-oncogenic or an inhibiting effect on tumor development
are denoted as proto-oncogenics and inhibitors, the former promotes tumor growth, the latter slows it down.

One such pathway is the Mitogen Activated Protein Kinase (MAPK) pathway. This pathway connects certain mutations of the
[BRAF](https://en.wikipedia.org/wiki/BRAF_(gene)) oncogene (i.e. DNA) to the generation of certain proteins that lead to the promotion of cell growth. This 
[MAPK pathway](https://en.wikipedia.org/wiki/MAPK/ERK_pathway) looks as follows:

![MAPK](_images/MAPKpathway_diagram.svg.png)

Just to show the complexity, the [PI3K/AKT/mTOR pathway](https://en.wikipedia.org/wiki/PI3K/AKT/mTOR_pathway) looks as follows:
![PI3k](_images/m836px-MTOR-pathway-v1.7.svg.png)

![mind blown!](_images/mind_blown.jpg)

Don't worry, this is not expected from us..although I can produce this stuff in paint, hands down (I can actually use my 
nose to draw..). Anyways, back to reality: we only have a few thousand protein measurements, and we do not have any
time series data so it is practically impossible to extract any feedback effects from downstream changes. 
Ooofff, that leaves us with a top down approach, from instruction/mutation to RNA and in some cases, proteins.
Finding any feedback effect is secondary, and perhaps for continued work after the hackathon.

Now, given such a pathway we can frustrate the signal anywhere on the chain, as long as it prevents
the cell growth stimulation.

We should keep in mind that the inhibitors/proto-oncogenes are likely specific to
the type of melanoma's, we distinguish at least the following by their genomic mutations:
* (proto-oncogenes) BRAF wild-type
* Triple Wild-type
* NF1 
* KIT
* MITF
* RAS
* (inhibitor) PD-1/PD-L1

The current inhibitor, the one they likely use in the immunotherapy is PD-L1. It is called a checkpoint inhibitor, retrieving this 
from our models would be a good validator. In fact, PD-L1 itself is a proto-oncogene, but apparently
it can be inhibited. So, instead of searching for inhibitors, we should be looking for
proto-oncogenes.

"Official" supporting questions:
* Can you show and visualize the correlations and concepts between the different datasets?
* As melanoma is a set of diverse diseases, can you stratify the patients based on all the data in to subgroups?
* Can you integrate all the data to make more accurate predictions for each patient than you would by only looking at one data source?
* Can you select a list of most informational variables that drive the predictions?
* Can you select a list of most informational variables distinctive for each patient subgroup?
* Can you identify a signature based on an integrative approach that can predict response to immunotherapy?
* Can you identify a signature that correlates with the prognosis of immunotherapy?

Basic hypotheses that would be nice to confirm
* T(tumor), increased Bresow-thickness correlates with more malignancy (Tis, T1a/b, T2a/b, T3a/b, T4a/b), i.e. decreasing survival rate
* N(nodal stage), Local spread correlates with more malignancy (N0, N1a/b, N2a/b/c, N3)
* M(metastasis location), distant metastasis (beyond regional lymph nodes)  corresponds with mmore malignancy (M0, M1)
* BRAF proto-oncogenic mutations should occcur in about 50% of all cutaneous melanomas.
* We should be able to identify 4 subtypes of cutaneous melanomas: BRAS/RAS(N/H/K)/NF1/Triple-WT
* order 3 clusters in the mRNA profiles of the most variant genes (keratin, immune, MITF-low)
* inhibitor: PD-L1/PD-1, our method should be able to retrieve this specific mutation as an inhibitor
* inhibitor: MEK, our method should be able to retrieve this specific mutation as an inhibitor for BRAF wild-type/NF1 mutant melanoma's
* inhibitor: PTEN/TP53/APC, our method should be able to retrieve this specific mutation as an inhibitor
* proto-oncogenic: BRAF, our method should be able to retrieve this specific mutation as a proto-oncogene
* LCK protein expression: correlates positively with patient survival

# Things biologists like

Hierarchical cluster diagrams, linked graphs, flow diagrams and simple tables/heatmaps with 
the most important genomes:

![Heat maps!](_images/ex_plot.png)

![Heat maps!](_images/ex_plot4.png)

![Importances](_images/ex_plot2.png)

![Importances](_images/ex_plot3.png)

![Circle](_images/ex_plot5.png)

![Circle](_images/ex_plot6.png)


Biologists like to understand the results, not very strange since they will base their laboratory 
work on it, medications will be derived from it and it will be applied to real patients. 
This is very important to keep in mind since it excludes a neural-network-only approach.

# Suggested approaches 

Please add ideas with your name in the section header.

## Classifications per layer 

This should be easy to do. First we should define the targets that are relevant to our end goal,
which is to recognize pathways, and inhibitors on those pathways. I.e. we need to be able to 
predict the level of malignancy, the survival rate and the response to immunotherapy.

This generates weights/importances per feature and gives the predictive power of each layer.

A novel thing to do is to apply a tool like Quiver to visualise which genomes are important 
per classification. For this we would need to transform the 1-dimensional tensors 
into 2-dimensional tensors. This would only work per patient, but as transparency 
is one of the key-ingredients of personalised medicine, and in my view of ML applied to
health care in general, it would be a nice touch. 

## Correlations between different layers, 

Extract layer pairs: DNA-RNA, RNA-mRNa, mRNa-proteomics, using 
* raw features
* raw, filtered features (using variance over the different classes)
* reduced features per layer (PCA/LDA whatever)

-> generate new cross-layers that combine all possible layer pairs into single layer,
train classifier and characterise the multilayer pairs.

I.e. Correlated features link the layers pairwise, after which the layers can be connected into 
a single layer. 


## Clusters per layer

We can try to find 
* communities
* exemplars 
* cliques

and then connect these cluster (proxies) between the layers.

## Bayesian Networks

Given potential pathways we can infer Bayesian Networks as approximations for the GRN and visualize them
with some graph viz. tool. 

## Graphs, from the ground up 

Per patient we have a graph connecting the genomes to RNA, to miRNA, etc. 
This graph will mostly be similar per patient in terms of the adjacency matrix 
but dissimilar in terms of the similarity matrix. This opens up some possibilities: 

* determine clusters per patient graph: exemplars, communities, cliques. Then determine cluster overlap
per target label.
* create multi-layer graph per target label, count edges (or sum edge weights), normalise edge sums. 
Flatten multi-layer graph and interpret normalised sums as edge weights. Determine characteristics clusters
per target label. I.e. (N, N, m) --> (N*, N*, 1)

The resulting clusters, and their characteristics can be used to feed a predictor. This has the benefit of 
* transparency: it is clear why a target value is predicted
* compatibility: compared to simply merging the data into one matrix we have more guarantee to obtain biologically
sound estimations

# Sources

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4731297/
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5496318/#MOESM8

https://www.nature.com/articles/nature13385

https://www.ncbi.nlm.nih.gov/pubmed/26091043
https://www.ncbi.nlm.nih.gov/pubmed/22960745