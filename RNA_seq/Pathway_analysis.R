library(knitr)
library(dplyr)
library(ggplot2)
library(stringr)
library(clusterProfiler)
library(org.Mm.eg.db)
library(AnnotationDbi)

# Utility functions
generate_genelist <- function(file_up, file_down){
  up_gene <- read.table(file_up, header = T, sep = "\t", row.names = NULL, stringsAsFactors = F)
  print(head(up_gene))
  print(dim(up_gene))
  
  down_gene <- read.table(file_down, header = T, sep = "\t", row.names = NULL,   stringsAsFactors = F)
  print(head(down_gene))
  print(dim(down_gene))
  
  all_gene <- rbind(up_gene, down_gene)
  print(head(all_gene))
  print(dim(all_gene))
  
  all_gene_no_dup <- all_gene[!duplicated(all_gene$gene),]
  geneList <- all_gene_no_dup[,c("log2FC")]
 
  names(geneList) <- as.character(all_gene_no_dup[,c("gene")])
  

  geneList <- sort(geneList, decreasing = TRUE)
  print(geneList[1:10])
  return(geneList)
}

convert_gene <- function(geneList, abs_thrshold){
  geneListNew <- names(geneList)[abs(geneList) > abs_thrshold]
  print(length(geneListNew))
  convert_id_gene <- select(org.Mm.eg.db, keys=geneListNew, columns=c("ENTREZID"), keytype="SYMBOL")
  print(head(convert_id_gene))
  print(nrow(convert_id_gene))
  return(convert_id_gene)
}

convert_gene_all <- function(geneList){
  geneListNew <- names(geneList)
  print(length(geneListNew))
  convert_id_gene <- select(org.Mm.eg.db, keys=geneListNew, columns=c("ENTREZID"), keytype="SYMBOL")
  convert_id_gene <- convert_id_gene[!duplicated(convert_id_gene$SYMBOL),]
  print(head(convert_id_gene))
  print(nrow(convert_id_gene))
  return(convert_id_gene)
}

run_KEGG <- function(gene, rds, kk_name_txt, kk_name_csv, rds_name){
  kk <- enrichKEGG(gene       = gene$ENTREZID,
                   organism     = 'mmu',
                   pvalueCutoff = 0.05,
                   keyType = "ncbi-geneid")
  print(head(kk))
  print(nrow(kk))
  saveRDS(kk, file = rds)
  kk_name <- setReadable(kk, 'org.Mm.eg.db', 'ENTREZID')
  print(head(kk_name))
  write.table(kk_name, kk_name_txt, quote = F, sep = "\t", col.names = T, row.names = F)
  write.table(kk_name, kk_name_csv, quote = F, sep = "\t", col.names = T, row.names = F)
  saveRDS(kk_name, file = rds_name)
  kk_com <- list(kk,kk_name)
  return(kk_com)
}

run_KEGG_gse <- function(gene, convert_gene, rds, kk_name_txt, kk_name_csv, rds_name){
  geneList_id <- gene
  names(geneList_id) <- as.character(convert_gene[,'ENTREZID'])
  kk_gse <- gseKEGG(geneList     = geneList_id,
                    organism     = 'mmu',
                    nPerm        = 1000,
                    minGSSize    = 120,
                    pvalueCutoff = 0.05,
                    verbose      = FALSE,
                    keyType = "ncbi-geneid")
  print(head(kk_gse))
  print(nrow(kk_gse))
  saveRDS(kk_gse, file = rds)
  kk_name <- setReadable(kk_gse, 'org.Mm.eg.db', 'ENTREZID')
  print(head(kk_name))
  write.table(kk_name, kk_name_txt, quote = F, sep = "\t", col.names = T, row.names = F)
  write.table(kk_name, kk_name_csv, quote = F, sep = "\t", col.names = T, row.names = F)
  saveRDS(kk_name, file = rds_name)
  kk_com <- list(kk_gse,kk_name)
  return(kk_com)
}

dot_plot <- function(pathway_obj, show_num, title, file_name, wid, hei){
  dotplot(pathway_obj, showCategory=show_num) + ggtitle(title)
  ggsave(file_name, width = wid, height = hei)
  #plot_grid(p1, p2, ncol=2)
}

net_plot <- function(gene, convert_gene, pathway_obj, file_name, wid, hei){
  geneList_id <- gene
  names(geneList_id) <- as.character(convert_gene[,'ENTREZID'])
  cnetplot(pathway_obj, categorySize="pvalue", foldChange=geneList_id)
  ggsave(file_name, width = wid, height = hei)
}

bar_plot <- function(pathway_obj, show_num, title, file_name, wid, hei){
  barplot(pathway_obj, showCategory=show_num) + ggtitle(title)
  ggsave(file_name, width = wid, height = hei)
  #plot_grid(p1, p2, ncol=2)
}

# Run KEGG
gene_list <- generate_genelist("up_gene_p0.05.txt",
                               "down_gene_p0.05.txt")
converted_list <- convert_gene(gene_list, 1)
converted_list_all <- convert_gene_all(gene_list)

kk_com <- run_KEGG(converted_list, 
                   "cond_p0.05/KEGG_logFC>1.rds", 
                   "cond_p0.05/KEGG_logFC>1.txt",
                   "cond_p0.05/KEGG_logFC>1.csv",
                   "cond_p0.05/KEGG_logFC>1_readable.rds")

kk_1 <- kk_com[[1]]
kk_name_1 <- kk_com[[2]]

kk_com <- run_KEGG(converted_list_all, 
                   "cond_p0.05/KEGG_all.rds", 
                   "cond_p0.05/KEGG_all.txt",
                   "cond_p0.05/KEGG_all.csv",
                   "cond_p0.05/KEGG_all_readable.rds")

kk_all <- kk_com[[1]]
kk_name_all <- kk_com[[2]]

bar_plot(kk_1, 8, "KEGG pathway analysis\n(log2FC>1 Exp vs Ctrl", "cond_p0.05/KEGG_log2FC>1_bar.pdf", 7,4)

