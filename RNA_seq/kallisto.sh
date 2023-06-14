#### Load modules
source /home/yf96/seq/sickle-1.33/sickle.sh
module load fastx/0.0.13
module load fastqc/0.10.1
module load python/2.7.12
module load cutadapt/1.14
module load fastx/0.0.13
module load samtools/1.3.1
module load kallisto/0.45.1

#### Set variables
## prefix for output files
prefix=${1}
fq1File=${2}
fq2File=${3}
indexFile=${4}
gtfFile=${5}
chromSize=${6}

echo "prefix : " $prefix

#### Gunzip or bunzip files
cp ${fq1File} .
cp ${fq2File} .
gunzip -fv *.fastq.gz 

#### Generate fastqc output
echo "Running fastqc on Read1..."
mkdir $prefix.R1.fastqc/
fastqc -o $prefix.R1.fastqc/ $prefix.R1.fastq 2> $prefix.R1.fqc.log.txt

echo "Running fastqc on Read2..."
mkdir $prefix.R2.fastqc/
fastqc -o $prefix.R2.fastqc/ $prefix.R2.fastq 2> $prefix.R2.fqc.log.txt


#### Filter reads on quality using sickle
echo "Filtering reads on quality..."
sickle pe -f $prefix.R1.fastq -r $prefix.R2.fastq -t sanger \
-o $prefix.filtered_R1.fq -p $prefix.filtered_R2.fq \
-s $prefix.filtered_singles.fq &> $prefix.sickle.log.txt
rm $prefix.filtered_singles.fq


#### Computes equivalence classes for reads and quantifies abundances
kallisto quant -i ${indexFile} -o ${prefix}.kallisto \
--genomebam -g ${gtfFile} -c ${chromSize} $prefix.filtered_R1.fq $prefix.filtered_R2.fq


### rm files
rm $prefix.R1.fastq $prefix.R2.fastq

### gzip output files
gzip *.fq
cd ${prefix}.kallisto
gzip *.bam
