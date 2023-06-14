library(dplyr)
library(stringr)


## file names
file1 <- "mpileup_in_ocr.txt"
file2 <- "mpileup_NOD_B6_in_snp_indel_extract_named.sorted.txt"
file3 <- "mpileup_NOD_B6_in_snp_indel_with_ocr.txt"
file4 <- "snp_indel_ocr_stat.txt"

inters <- read.table(file1, header = F, sep = "\t", 
                     col.names = c("chrA", "startA", "endA","nameA", "chrB", "startB", "endB","nameB"),row.names = NULL,
                     stringsAsFactors = FALSE)

namedf <- read.table(file2, header = T, sep = "\t",row.names = NULL, stringsAsFactors = FALSE)
namedf_new <- filter(namedf, snpindelID %in% unique(inters$nameA))


namedf_new <- arrange(namedf_new, snpindelID)
inters <- arrange(inters, nameA)
inters_count <- inters %>% group_by(nameA) %>% summarise(count=n()) %>% arrange(nameA)


lineNum <- rep(c(1:nrow(namedf_new)), times=inters_count$count)
namedf_dup <- namedf_new[lineNum,]
sum(namedf_dup[,c("snpindelID")]!=inters[,c("nameA")])


namedf_dup$ocr <- inters$nameB
write.table(namedf_dup, file3, quote = F, sep = "\t", col.names = T, row.names = F)

snp_indel_ocr_stat <- namedf_dup %>% group_by(ocr) %>% summarise(snp_indel_count=n() ,B6_count=sum(B6),
                                                                   NOD_count=sum(NOD))
write.table(snp_indel_ocr_stat, file4, quote = F, sep = "\t", col.names = T, row.names = F)

