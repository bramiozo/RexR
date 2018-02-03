#!/usr/bin/env Rscript

### Empty the variables
rm(list=ls(all=TRUE))
options(stringsAsFactors = FALSE)

# Load the libraries


# Input data
data.read <- read.table("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/BatchCorrection/Cohort1_ALL10_RMA_ComBat.txt",
                        sep="\t",header=T, row.names=1, check.names = FALSE)
sample.vars <- read.table("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/AdditionalFiles/PatientCharacteristicsHR_vs_NHR_ALL10.txt", 
                          header=T, check.names = FALSE)
setwd("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/WilcoxonFDR")


# Display data if desired
#dim(data.read)
#colnames(data.read)

# Retrieve subset containing all ALL10 patients as marked in sample.vars (HR vs NHR)
ALL10_Patients <- data.read[, (names(data.read) %in% rownames(sample.vars))]

my.class <- 1*(sample.vars$HR==1) + 2*(sample.vars$NHR==1)
table(my.class,sample.vars$HR)


# Perform the tests
source("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/Scripts/functions_wilcoxon_fdr_braintumors.txt")
pdf("ALL10_wpvals_HR_versus_NHR.pdf",width=8,height=6)
wpvals.HRvsNHR <- wpvals.pairs(ALL10_Patients,my.class,2)
adj.wpvals.HRvsNHR <- adj.bhfdr(wpvals.HRvsNHR)
dev.off()
save.image("ALL10_normalized_HRvsNHR_analyse.RData ")


source("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/Scripts/functions_wilcoxon_fdr_braintumors.txt")
table.npvals(adj.wpvals.HRvsNHR)
#all.TCRpreT <- sort(adj.wpvals.TCRpreT[54675])Deze alleen als je een selectie van bijvoorbeeld de top100 wil doen, voor de totale lijst is deze niet nodig
write.table(list(rownames(data.read),wpvals.HRvsNHR,adj.wpvals.HRvsNHR),
            file="ALL10_HRvsNHR.txt",sep="\t",col.names=c("Probe ids","Wilcoxon pvalues","FDR"),
            row.names=F)


# Get all the significant genes
HRvsNHR = read.table(sep = "\t", row.names = 1, header = TRUE, "ALL10_HRvsNHR.txt", 
                     check.names = FALSE)
SignificantSet <- data.frame(HRvsNHR[HRvsNHR$FDR < 0.05, ])
# Rename the identifiers
SignificantSet <- setNames(cbind(rownames(SignificantSet), SignificantSet, row.names = NULL), 
         c("Probe ids","Wilcoxon pvalues","FDR"))
write.table(SignificantSet, file="ALL10_HRvsNHR_Significant.txt", sep="\t", col.names = TRUE,
            row.names = FALSE)


### Get a subset of RMA data
ALL10.data <- read.table("/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/BatchCorrection/ALL10_subset_RMA_ComBat.txt",
                         sep="\t",header=T, row.names=1, check.names = FALSE)

Sig.ALL10 <- ALL10.data[rownames(ALL10.data) %in% SignificantSet$`Probe ids`,]
write.table(Sig.ALL10,file="/Users/rhagelaar/Desktop/T_ALL_Data/ResultFolder/WilcoxonFDR/ALL10_significant_subset_RMA_ComBat.txt", 
            sep="\t",col.names=NA)
