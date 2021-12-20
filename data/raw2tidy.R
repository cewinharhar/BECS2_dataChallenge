library(dplyr)
library(data.table) #ultra fast data read in
library(stringr)
library(tidyr)


############################################
uniprotScrape <- read.delim("20211129-144739_proteinSeqInfoExt.csv", sep=",", row.names = 1) %>% 
  subset(select = -c(organism))

metadata <- read.delim("metadata.csv", sep="\t")



##---------------------------------------------------------------------------------------##
## -----------------------------   Depleted PREPARATION    ----------------------------- ##
##---------------------------------------------------------------------------------------##
depleted_raw <- fread("S:/Ana/2021/821_Depletion_Panel_Designer_DPD/IP_726/depleted_report/20211027_124634_ip726_depleted/Depleted_Report_EDA_KMY.xls", sep="\t") %>%
  transform(PG.MolecularWeight = as.numeric(PG.MolecularWeight)) %>%  #molecular weight to numeric
  separate_rows(PG.UniProtIds, sep=";")  %>% #seperate multiple uniprot entries 
      arrange(desc(PG.Quantity)) 

depleted_raw$R.FileName = depleted_raw$R.FileName %>% 
  str_replace_all("_X01", "")

#Sample which are pooled. They do not appear in native sample set therefore are beeing removed
pool <- c("G_D200423_MDIA_P671_S44_R01", 
          "G_D200423_MDIA_P671_S44_R02",
          "G_D200423_MDIA_P671_S44_R03",
          "G_D200429_MDIA_P671_S02_R01",
          "G_D200429_MDIA_P671_S02_R02",
          "G_D200429_MDIA_P671_S02_R03",
          "G_D200423_MDIA_P671_S53_R01",
          "G_D200429_MDIA_P671_S11_R01",
          "G_D200423_MDIA_P671_S64_R01",
          "G_D200429_MDIA_P671_S22_R01",
          "G_D200423_MDIA_P671_S73_R01",
          "G_D200429_MDIA_P671_S31_R01",
          "G_D200423_MDIA_P671_S43_R01",
          "G_D200429_MDIA_P671_S01_R01",
          "G_D200423_MDIA_P671_S45_R01",
          "G_D200429_MDIA_P671_S03_R01")

#filter non-spike-in and remove samples from pool 
depleted <- subset(depleted_raw, EG.Workflow != "SPIKE_IN" & !(R.FileName %in% pool))     # remove SPIKE-IN Proteins 

#join metadata and depleted
depleted <- left_join(depleted, metadata, by=c("R.FileName" = "depletedID")) %>%
  subset(select = -c(nativeID))

names(depleted)[[7]] <- "uniprot"
names(depleted)[[6]] <- "ProteinName"
depleted$status <- "depleted"

depleted= subset(depleted, Column != 1) %>%
  left_join(uniprotScrape, by=c("uniprot"))

#######################################################################

summaryAll = depleted%>%
  dplyr::group_by(R.FileName, uniprot, proteinName, Group, fastaSequence) %>%
  dplyr::summarise(meanAbu=mean(PG.Quantity),
                   log10meanAbu = log10(mean(PG.Quantity)),
                   stdDev = sd(PG.Quantity),
                   relStdDev = sd(PG.Quantity) / mean(PG.Quantity)) %>%
                   as.data.frame()

summaryAll = summaryAll %>%
  group_by(Group) %>%
    dplyr::mutate(Rank=rank(-meanAbu, na.last="keep"))
                   
fwrite(summaryAll, file= "becs2.csv")

################3 comparison healthy tumor  ###########################

# Not finished yet 

vulcano_group = depleted %>%
  dplyr::group_by(uniprot, proteinName, Group) %>%
  dplyr::summarise(meanAbu = mean(PG.Quantity),
                   log10meanAbu = log10(mean(PG.Quantity)),
                   foldChange=mean(PG.Quantity[status=="depleted"]) / mean(PG.Quantity[status=="native"]),
                   Log2FoldChange=log2(mean(PG.Quantity[status=="depleted"]) / mean(PG.Quantity[status=="native"])),
                   stdDev = sd(PG.Quantity),
                   relStdDev = sd(PG.Quantity) / mean(PG.Quantity),
                   pvalueC1=tryCatch({t.test(PG.Quantity[status=="native"], PG.Quantity[status=="depleted"])$p.value},error=function(i){NA})) %>%
  dplyr::mutate(Rank=rank(-meanAbu, na.last="keep"))

vulcano_group = vulcano_group %>%
  left_join(plasmaprot[c("uniprot", "conc_thpa_ugl", "mars14", "sepromix20")], by=c("uniprot")) %>%
  replace_na(list(mars14=0, sepromix20=0)) %>%
  arrange(Rank) %>% 
  as.data.frame()

vulcano_group = vulcano_group %>%
  dplyr::mutate(RankFC = rank(Log2FoldChange, na.last="keep")) %>% #Make rank for log2(foldchange) and na.last = keep --> NA instead of rank nr for na's
  dplyr::mutate(RankQ = ntile(desc(RankFC),60))   


fwrite(vulcano_group, "vulcano_group.csv")
