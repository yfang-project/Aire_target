#### Load modules
module load gcc/6.2.0  

#### Set variables
## prefix for output files
prefix=${1}
## working dir
workDir=${2}
genomeDir=${3}


#### Change to working directory
cd ${workDir}
echo "Current working directory: ${workDir}"

#### Load modules
module load python/3.6.0  
source /home/yf96/python_env/umitools/bin/activate

umi_tools extract --bc-pattern="^(?P<umi_1>.{8})(?P<cell_1>.{8})" \
--extract-method=regex \
--stdin $prefix.trim1.R1.fq \
--stdout $prefix.extracted.R1.fq \
--log=$prefix.processed.log

deactivate
module unload python/3.6.0

#### Load modules  
module load fastx/0.0.13
module load fastqc/0.10.1
module load bowtie2/2.3.4.3
module load samtools/1.3.1
module load bedtools/2.27.1
module load picard/2.8.0

#### Align reads using bowtie2
# -p = number of threads to use 
# -x = location of indexed genome
# -S = name of .sam output file
echo "Mapping reads to mm10..."
bowtie2 -p 8 -x ${genomeDir} -X 1000 -U $prefix.extracted.R1.fq -S $prefix.btout2.sam


#### Filter out reads that didn't map, keep only mapped
## -h = include the header in the output
## -S = input is in sam format
## -F = skip alignments with bits present in INT [0], so F 4 = 100 = 1 = T in position testing unmapped? so skip these?
samtools view -hS -F 4 $prefix.btout2.sam > $prefix.mapped.sam


#### Keep only reads that mapped to single best location
sed '/XS:/d' $prefix.mapped.sam > $prefix.mapped_1alignmentonly.sam
# generate bam version and sort it
samtools view -bhS $prefix.mapped_1alignmentonly.sam > $prefix.mapped_1align.bam
samtools sort -o $prefix.mapped_1align.sorted.bam $prefix.mapped_1align.bam
samtools index $prefix.mapped_1align.sorted.bam

#### Load modules
module load python/3.6.0  
source /home/yf96/python_env/umitools/bin/activate

umi_tools dedup -I $prefix.mapped_1align.sorted.bam --output-stats=deduplicated -S $prefix.sorted.uniq.bam

deactivate
module unload python/3.6.0


#### Filter out reads mapping to unassembled genome
# chrUn, random
samtools index $prefix.sorted.uniq.bam

samtools idxstats $prefix.sorted.uniq.bam | cut -f 1 | \
egrep -v 'random|chrUn' | xargs samtools view \
-b $prefix.sorted.uniq.bam > $prefix.sorted.uniq.stdchroms.bam


#### Filter out reads mapped to "chrM"
samtools index $prefix.sorted.uniq.stdchroms.bam

samtools idxstats $prefix.sorted.uniq.stdchroms.bam | cut -f 1 | \
egrep -v 'chrM' | xargs samtools view \
-b $prefix.sorted.uniq.stdchroms.bam > $prefix.sorted.uniq.nomito.bam 

## samtools index $prefix.sorted.uniq.nomito.bam
## it will be indexed in the qc report step
 

### gzip output files
echo "Gzipping output files..."
gzip *.fq
gzip *.bam


echo "Finished!"

