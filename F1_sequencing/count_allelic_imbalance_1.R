library(dplyr)
library(stringr)

## file names
file1 <- "mpileup_NOD_B6_in_snp_indel.sorted.txt"
file2 <- "mpileup_NOD_B6_in_snp_indel_extract.sorted.txt"
file3 <- "mpileup_NOD_B6_in_snp_indel_extract_named.sorted.txt"
file4 <- "mpileup_NOD_B6_in_snp_indel.bed"
  
mpile_snp_indel <- read.table(file1, header = F, sep = "\t", 
                              col.names = c("CHROM",	"POS","ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "NOD", "B6"), 
                              row.names = NULL, comment.char = "#", stringsAsFactors = F)

mpile_snp_indel_uniq <-mpile_snp_indel[!duplicated(mpile_snp_indel[,c("CHROM", "POS", "REF")]), 
                                       c("CHROM", "POS", "POS","REF","NOD", "B6")]
colnames(mpile_snp_indel_uniq) <- c("chr", "start", "end","REF","NOD", "B6")

B6_num <- mpile_snp_indel_uniq$B6
B6_num_new <- str_extract(B6_num, ":[0-9]+:")
B6_num_new <- gsub(":", "", B6_num_new)
B6_num_new <- as.numeric(B6_num_new)
length(B6_num_new)
mpile_snp_indel_uniq$B6 <- B6_num_new

NOD_num <- mpile_snp_indel_uniq$NOD
NOD_num_new <- str_extract(NOD_num, ":[0-9]+:")
NOD_num_new <- gsub(":", "", NOD_num_new)
NOD_num_new <- as.numeric(NOD_num_new)
length(NOD_num_new)
mpile_snp_indel_uniq$NOD <- NOD_num_new

write.table(mpile_snp_indel_uniq, file2, quote = F, sep = "\t", col.names = T, row.names = F)

mpile_snp_indel_uniq_named <- mpile_snp_indel_uniq[,]
mpile_snp_indel_uniq_named$snpindelID <- paste0("snpindelID_", mpile_snp_indel_uniq_named$chr,
                                          "_", mpile_snp_indel_uniq_named$start, "_", mpile_snp_indel_uniq_named$REF)


write.table(mpile_snp_indel_uniq_named, file3, quote = F, sep = "\t", col.names = T, row.names = F)

mpile_snp_indel_uniq_bed <- mpile_snp_indel_uniq_named[,c("chr", "start", "end", "snpindelID")]
write.table(mpile_snp_indel_uniq_bed, file4, quote = F, sep = "\t", col.names = F, row.names = F)

