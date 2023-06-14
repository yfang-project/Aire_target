#### Load modules
module load gcc/6.2.0  
module load homer/4.9
module load ucsc-tools/363
module load bedtools/2.27.1
module load samtools/1.3.1


#### Set variables
PREFIX=${1}
echo "prefix : " $PREFIX
blackListFile=${2}
workDir=${3}
bamDir=${4}
genomeDir=${5}


#### COPY needed files to the working directory
echo "Copy bam bai file into the current directory"
cd ${bamDir}
cp *.sorted.uniq.nomito.bam.gz ${workDir}
cp *.sorted.uniq.nomito.bam.bai ${workDir}

echo "Go back to the working dir:"
echo "${workDir}"
cd ${workDir}

#### Gunzip files
echo "Decompressing files..."
gunzip -fv *.bam.gz


#### makeTagDirectory and generate QC files
echo "Making Tag Directory..."
makeTagDirectory $PREFIX.homer.tagdir/ -genome ${genomeDir} \
-tbp 1 -checkGC -illuminaPE *sorted.uniq.nomito.bam &> $PREFIX.tagdir.log.txt


#### Find peaks
## -style		settings for peak finding based on transcription factor, histone, dnase
## -o			  output file name,
## -norm			# of reads to normalize to, default = 1e7 changed to 1e6
echo "Finding peaks in factor mode..."
findPeaks $PREFIX.homer.tagdir/ -style factor -L 2 -LP 0.0001 -F 1.5 -poisson 0.00001 \
-o $PREFIX.peaks.factor.txt &> $PREFIX.log.peaks.factor.txt

echo "Finding peaks in dnase mode..."
findPeaks $PREFIX.homer.tagdir/ -style dnase -L 2 -LP 0.0001 -F 1.5 -poisson 0.00001 \
-o $PREFIX.peaks.dnase.txt &> $PREFIX.log.peaks.dnase.txt

echo "Finding peaks in histone mode..."
findPeaks $PREFIX.homer.tagdir/ -style histone -F 1.5 -poisson 0.00001 \
-o $PREFIX.peaks.histone.txt &> $PREFIX.log.peaks.histone.txt

## get into bed format
sed '/^#/ d' $PREFIX.peaks.factor.txt | cut -f 2-4  > $PREFIX.peaks.factor.bed
sed '/^#/ d' $PREFIX.peaks.dnase.txt | cut -f 2-4  > $PREFIX.peaks.dnase.bed
sed '/^#/ d' $PREFIX.peaks.histone.txt | cut -f 2-4  > $PREFIX.peaks.histone.bed

echo "Merging results of histone and factor modes..."
mergePeaks $PREFIX.peaks.histone.txt $PREFIX.peaks.factor.txt > $PREFIX.peaks.bliss.txt
## get into bed format
cut -f 2-4 $PREFIX.peaks.bliss.txt | sed '1d' > $PREFIX.peaks.bliss.bed

#### Remove ENCODE blacklist regions for mm10
echo "Removing ENCODE blacklist regions for mm10..."
bedtools subtract -A -a $PREFIX.peaks.factor.bed -b ${blackListFile} > $PREFIX.peaks.factor.cleaned.bed
bedtools subtract -A -a $PREFIX.peaks.histone.bed -b ${blackListFile} > $PREFIX.peaks.histone.cleaned.bed
bedtools subtract -A -a $PREFIX.peaks.dnase.bed -b ${blackListFile} > $PREFIX.peaks.dnase.cleaned.bed
bedtools subtract -A -a $PREFIX.peaks.bliss.bed -b ${blackListFile} > $PREFIX.peaks.bliss.cleaned.bed

#### Remove bam files
echo "Remove bam files..."
rm *.bam
rm *.bai
