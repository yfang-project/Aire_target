#### Load modules
module load fastx/0.0.13
module load fastqc/0.10.1
module load python/2.7.12
module load cutadapt/1.14
module load fastx/0.0.13
module load bowtie2/2.3.4.3
module load samtools/1.3.1
module load bedtools/2.27.1
module load picard/2.8.0

#### Set variables
## prefix for output files
prefix=${1}
## working dir
workDir=${2}
genomeDir=${3}


#### Change to working directory
cd ${workDir}
echo "Current working directory: ${workDir}"

#### Gunzip or bunzip files
echo "Decompressing fq files..."
files=$(ls ./*.bz2 2> /dev/null | wc -l)
if [ "$files" != 0 ]; then
  bunzip2 -fv *R1.fq.bz2 *R2.fq.bz2
else
  gunzip -fv *R1.fq.gz *R2.fq.gz
fi


#### Generate fastqc output
echo "Running fastqc on Read1..."
mkdir $prefix.R1.fastqc/
fastqc -o $prefix.R1.fastqc/ $prefix.R1.fq 2> $prefix.R1.fqc.log.txt

#### Generate fastqc output
echo "Running fastqc on Read2..."
mkdir $prefix.R2.fastqc/
fastqc -o $prefix.R2.fastqc/ $prefix.R2.fq 2> $prefix.R2.fqc.log.txt

#### Filter reads on quality using sickle
echo "Filtering reads on quality..."
sickle pe -f $prefix.R1.fq -r $prefix.R2.fq -t sanger \
-o $prefix.filtered_R1.fq -p $prefix.filtered_R2.fq \
-s $prefix.filtered_singles.fq &> $prefix.sickle.log.txt
rm $prefix.filtered_singles.fq

#### Clip adapters from sequences using cutadapt
## e = maximum error rate, default = 0.1
## m = minimum length, throw away reads shorter than N bases
echo "Clipping adapter from 5' side..."
cutadapt -g AGATGTGTATAAGAGACAG -G CTGTCTCTTATACACATCT -e 0.1 -m 20 \
-o $prefix.trim1_R1.fq -p $prefix.trim1_R2.fq \
$prefix.filtered_R1.fq $prefix.filtered_R2.fq

echo "Clipping adapter from 3' side..."
cutadapt -a AGATGTGTATAAGAGACAG -A CTGTCTCTTATACACATCT -e 0.1 -m 20 \
-o $prefix.trim2_R1.fq -p $prefix.trim2_R2.fq \
$prefix.trim1_R1.fq $prefix.trim1_R2.fq


#### Align reads using bowtie2
# -p = number of threads to use 
# -x = location of indexed genome
# -S = name of .sam output file
echo "Mapping reads to mm10..."
bowtie2 --local --very-sensitive --no-mixed --no-discordant --phred33 -I 10 -X 1000 --fr -p 8 -x ${genomeDir} -1 $prefix.trim2_R1.fq -2 $prefix.trim2_R2.fq \
-S $prefix.btout2.sam &> ${prefix}_bowtie2.txt


#### Filter out reads that didn't map, keep only mapped
## -h = include the header in the output
## -S = input is in sam format
## -F = skip alignments with bits present in INT [0], so F 4 = 100 = 1 = T in position testing unmapped? so skip these?
samtools view -hS -F 4 $prefix.btout2.sam > $prefix.mapped.sam

#### Keep only mate pairs 
## -f = keep alignments with bits present in INT, so 0x2 = read mapped in proper pair?
samtools view -hS -f 0x2 $prefix.mapped.sam > $prefix.mapped.mates.sam

#### Keep only reads that mapped to single best location
sed '/XS:/d' $prefix.mapped.mates.sam > $prefix.mapped_1alignmentonly.sam
# generate bam version and sort it
samtools view -bhS $prefix.mapped_1alignmentonly.sam > $prefix.mapped_1align.bam
samtools sort -o $prefix.mapped_1align.sorted.bam $prefix.mapped_1align.bam

#### Filter out duplicates
java -Xms1024m -jar $PICARD/picard-2.8.0.jar MarkDuplicates INPUT=$prefix.mapped_1align.sorted.bam \
OUTPUT=$prefix.sorted.uniq.bam METRICS_FILE=${prefix}_picard.rmDup.txt REMOVE_DUPLICATES=true 2> $prefix.picard.log.txt

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


### gzip output files
echo "Gzipping output files..."
gzip *.fq
gzip *.bam
gzip *.sam


echo "Finished!"
