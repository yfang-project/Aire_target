#### Load modules
module load fastx/0.0.13
module load fastqc/0.10.1
module load python/2.7.12
module load cutadapt/1.14
module load fastx/0.0.13
module load samtools/1.3.1
module load bowtie2/2.3.4.3
module load tophat/2.1.1
module load bedtools/2.27.1
module load picard/2.8.0

#### Set variables
## prefix for output files
prefix=${1}
genomeDir=${2}
genomeName=${3}
workDir=${4}
fastqdir=${5}

echo "prefix : " $prefix

#### cp files into the working dir
cd ${workDir}
cp ${fastqdir}/${prefix}* .

#### Gunzip files
echo "Decompressing fq files..."
gunzip -fv *R1.fastq.gz *R2.fastq.gz


#### Generate fastqc output
echo "Running fastqc on Read1..."
mkdir $prefix.R1.fastqc/
fastqc -o $prefix.R1.fastqc/ *R1.fastq 2> $prefix.R1.fqc.log.txt

echo "Running fastqc on Read2..."
mkdir $prefix.R2.fastqc/
fastqc -o $prefix.R2.fastqc/ *R2.fastq 2> $prefix.R2.fqc.log.txt


#### Filter reads on quality using sickle
echo "Filtering reads on quality..."
sickle pe -f *R1.fastq -r *R2.fastq -t sanger \
-o $prefix.filtered_R1.fq -p $prefix.filtered_R2.fq \
-s $prefix.filtered_singles.fq &> $prefix.sickle.log.txt
rm $prefix.filtered_singles.fq


#### Align reads using tophat
echo "Mapping reads..."
tophat -p 8 --segment-length 18 --no-coverage-search -o ./tophat.out ${genomeDir} $prefix.filtered_R1.fq $prefix.filtered_R2.fq

cd tophat.out

cp accepted_hits.bam ${workDir}/$prefix.accepted_hits.bam

cd ${workDir}

#### Filter out reads that didn't map, keep only mapped
samtools view -h -F 4 $prefix.accepted_hits.bam > $prefix.mapped.sam


#### Keep only mate pairs 
## -f = keep alignments with bits present in INT, so 0x2 = read mapped in proper pair?
samtools view -hS -f 0x2 $prefix.mapped.sam > $prefix.mapped.mates.sam


# generate bam version and sort it
samtools view -bhS $prefix.mapped.mates.sam > $prefix.mapped.mates.bam
samtools sort -o $prefix.mapped.mates.sorted.bam $prefix.mapped.mates.bam


#### Filter out reads mapping to unassembled genome
# chrUn, random
samtools index $prefix.mapped.mates.sorted.bam

samtools idxstats $prefix.mapped.mates.sorted.bam | cut -f 1 | \
egrep -v 'random|chrUn' | xargs samtools view \
-b $prefix.mapped.mates.sorted.bam > $prefix.sorted.uniq.stdchroms.bam


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

