#### Load modules
module load gcc/6.2.0  
module load python/2.7.12
module load macs2/2.1.1.20160309
module load bedtools/2.27.1
module load samtools/1.3.1


#### Set command parameters
REP1=${1}
REP2=${2}
GenomeSize=${3}
workDir=${4}
bam1Dir=${5}
bam2Dir=${6}
OUTP=${7}
PTHRES=${8}
IDRTHRES=${9}
blackListFile=${10}

#### COPY needed files to the working directory
echo "Copy bam bai file of rep1 into the current directory"
cd ${bam1Dir}
cp *.sorted.uniq.nomito.bam.gz ${workDir}
cp *.sorted.uniq.nomito.bam.bai ${workDir}

echo "Copy bam bai file of rep2 into the current directory"
cd ${bam2Dir}
cp *.sorted.uniq.nomito.bam.gz ${workDir}
cp *.sorted.uniq.nomito.bam.bai ${workDir}

echo "Go back to the working dir:"
echo "${workDir}"
cd ${workDir}

#### Gunzip files
echo "Decompressing files..."
gunzip -fv *.bam.gz

#### Merge BAMs of rep
echo "Merge bam files of rep..."
samtools merge ${OUTP}_merge.sorted.uniq.nomito.bam ${REP1}.sorted.uniq.nomito.bam ${REP2}.sorted.uniq.nomito.bam
samtools index ${OUTP}_merge.sorted.uniq.nomito.bam


#### call peaks on replicates and merged BAM and filter out ENCODE blacklist regions for mm10
echo "Call peaks on replicates and merged BAM and filter out ENCODE blacklist regions for mm10"
for PEAKBAM in ${OUTP}_merge.sorted.uniq.nomito.bam ${REP1}.sorted.uniq.nomito.bam ${REP2}.sorted.uniq.nomito.bam
do
    PEAKN=$(echo "$PEAKBAM" | awk -F'[.]' '{print $1}')
    macs2 callpeak -g ${GenomeSize} --keep-dup all -p ${PTHRES} -n ${PEAKN}_cutsite -t ${PEAKBAM}
    cat ${PEAKN}_cutsite_peaks.narrowPeak | cut -f 1-3 > ${PEAKN}_cutsite.uncleaned.peaks.bed
    bedtools subtract -A -a ${PEAKN}_cutsite.uncleaned.peaks.bed -b ${blackListFile} > ${PEAKN}_cutsite.cleaned.peaks.bed
done

module unload python/2.7.12
module load python/3.6.0
module load idr/2.0.2


#### filter peaks based on IDR and filter out ENCODE blacklist regions for mm10
echo "Running idr to filter peaks..."
idr --samples ${REP1}_cutsite_peaks.narrowPeak ${REP2}_cutsite_peaks.narrowPeak --peak-list ${OUTP}_merge_cutsite_peaks.narrowPeak --input-file-type narrowPeak --rank p.value --output-file ${OUTP}_cutsite.idr --soft-idr-threshold ${IDRTHRES} --plot --log-output-file ${OUTP}_cutsite.idr.log
IDRCUT=`echo "-l(${IDRTHRES})/l(10)" | bc -l`
echo "idr cut: ${IDRCUT}"
cat ${OUTP}_merge_cutsite_peaks.narrowPeak | grep -w -Ff <(cat ${OUTP}_cutsite.idr | awk '$12>='"${IDRCUT}"'' | cut -f 1-3) > ${OUTP}_cutsite.filtered.peaks 
cat ${OUTP}_cutsite.filtered.peaks | cut -f 1-3 > ${OUTP}_cutsite.filtered.peaks.bed

bedtools subtract -A -a ${OUTP}_cutsite.filtered.peaks.bed -b ${blackListFile} > ${OUTP}_cutsite.filtered.cleaned.peaks.bed
gzip ${OUTP}_cutsite.idr


#### Remove bam files
echo "Remove bam files..."
rm *.bam
rm *.bai

