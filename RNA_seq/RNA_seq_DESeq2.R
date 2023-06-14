library(DESeq2)
library(ggplot2)
library(tximport)
library(dplyr)
library(pheatmap)
library(RColorBrewer)
library(apeglm)
library(preprocessCore)
library(purrr)
library(matrixStats)
library(reshape2)

# Load the gene count table
cts <- read.delim('Genes_count_table.tsv', header = TRUE, sep = '\t', row.names = 1, stringsAsFactors = F)
cts <- as.matrix(cts)
print(head(cts))

# Run DEseq2
sampleTable <- data.frame(condition = factor(rep(c("Ctrl", "Exp"), c(3,3))))
rownames(sampleTable) <- colnames(cts)

dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = sampleTable,
                              design = ~ condition)

keep <- rowSums(counts(dds)) >= 6
dds <- dds[keep,]
dds$condition <- relevel(dds$condition, ref = "Ctrl")
dds_deseq <- DESeq(dds)


deseq_analysis <- function(dds_deseq, padjv, log2FC, up_name, down_name){
  res <- results(dds_deseq, alpha=padjv, lfcThreshold=log2FC, filter = rep(1, nrow(dds_deseq)))
  up <- (res$padj <= padjv) & (res$log2FoldChange >= log2FC)
  up[is.na(up)] <- FALSE
  up_gene <- res[up,]
  
  down <- (res$padj <= padjv) & (res$log2FoldChange <= -log2FC)
  down[is.na(down)] <- FALSE
  down_gene <- res[down,]
  
  up_gene <- up_gene[order(up_gene$pvalue),]
  down_gene <- down_gene[order(down_gene$pvalue),]
  
  print("Number of upregulated genes:")
  print(nrow(up_gene))
  print("Number of downregulated genes:")
  print(nrow(down_gene))
  
  write.table(data.frame(gene=up_gene@rownames, mean=up_gene$baseMean, log2FC=up_gene$log2FoldChange, pvalue=up_gene$pvalue,
                         padj=up_gene$padj, lfcSE=up_gene$lfcSE, stat=up_gene$stat), file = up_name, row.names = FALSE, quote = FALSE
              , sep = "\t")
  write.table(data.frame(gene=down_gene@rownames, mean=down_gene$baseMean, log2FC=down_gene$log2FoldChange, pvalue=down_gene$pvalue,
                         padj=down_gene$padj, lfcSE=down_gene$lfcSE, stat=down_gene$stat), file = down_name, row.names = FALSE, 
              quote = FALSE, sep = "\t")
}


deseq_analysis(dds_deseq, 0.1, 0,
               "Upgene.txt",
               "Downgene.txt")


res <- results(dds_deseq, filter = rep(1, nrow(dds_deseq)), alpha = 0.1)
summary(res)
write.table(data.frame(gene=res@rownames, mean=res$baseMean, log2.fold_change.=res$log2FoldChange,
                       p_value=res$pvalue, q_value=res$padj, lfcSE=res$lfcSE, stat=res$stat), 
            file = "ctrl_vs_exp_rna_stats.txt", 
            row.names = FALSE, quote = FALSE, sep = "\t")


plot_pca <- function(dds_deseq, file_name){
  vsd <- vst(dds_deseq, blind=TRUE)
  pcaData <- plotPCA(vsd, intgroup=c("condition"), returnData=TRUE)
  percentVar <- round(100 * attr(pcaData, "percentVar"))
  ggplot(pcaData, aes(PC1, PC2, color=condition)) +
    geom_point(size=3) +
    xlab(paste0("PC1: ",percentVar[1],"% variance")) +
    ylab(paste0("PC2: ",percentVar[2],"% variance")) + 
    theme_bw()
  ggsave(file_name, width = 5, height = 3)
}
plot_pca(dds_deseq, "ctrl_vs_exp_rna_pca.pdf")

plot_dist <- function(dds_deseq, file_name, coldata){
  vsd <- vst(dds_deseq, blind=TRUE)
  sampleDists <- dist(t(assay(vsd)))
  sampleDistMatrix <- as.matrix(sampleDists)
  #  rownames(sampleDistMatrix) <- vsd$condition
  rownames(sampleDistMatrix) <- rownames(coldata)
  colnames(sampleDistMatrix) <- NULL
  #colors <- colorRampPalette( rev(brewer.pal(9, "Blues")) )(255)
  pdf(file_name, width = 5, height = 3)
  pheatmap(sampleDistMatrix,
           clustering_distance_rows=sampleDists,
           clustering_distance_cols=sampleDists)
  dev.off()
}
plot_dist(dds_deseq, "ctrl_vs_exp_rna_dist.pdf", sampleTable)

# Plotting
res_df <- read.table('ctrl_vs_exp_rna_stats.txt', header = TRUE, sep = '\t', stringsAsFactors = F)
res_df <- res_df %>% arrange(p_value)

aig_df <- read.table("AIG_mm10_info.txt", header = T, row.names = NULL, sep = "\t", quote = "")
ang_df <- read.table("ANG_mm10_info.txt", header = T, row.names = NULL, sep = "\t", quote = "")

res_df_comb <- res_df %>% mutate(changed_gene = ifelse(gene %in% ang_df$gene_name, 'ANG',
                                                       ifelse(gene %in% ang_df$gene_name, 'AIG', 'Others')))

ggplot()+
  geom_point(data=res_df_comb%>% filter(changed_gene=='Others'), 
             mapping=aes(x=log2(mean), y=log2.fold_change.), size= 1, alpha=0.5, color='grey')+
  stat_density2d(data=res_df_comb %>% filter(changed_gene=='ANG'),
                 aes(x=log2(mean), y=log2.fold_change., fill = stat(level)), geom="polygon", alpha=0.3) +
  scale_fill_gradient(low = "lightskyblue1", high = "darkred", name='ANG density') +
  geom_hline(yintercept = 0, linetype='dotted')+
  theme_classic()+
  ylim(-4,4)+
  xlim(1,15)+
  xlab('Mean expression (log2)')+
  ylab('Exp / Ctrl (log2)')
ggsave(paste0("ma_plot.pdf"), width = 5, height = 3.5)


