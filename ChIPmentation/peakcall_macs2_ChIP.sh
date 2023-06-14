#### Load modules
module load gcc/6.2.0  
module load python/2.7.12
module load macs2/2.1.1.20160309
module load bedtools/2.27.1
module load samtools/1.3.1


#### Set command parameters
REP1=${1}
REP2=${2}
ctrl=${3}
GenomeSize=${4}
workDir=${5}
OUTP=${6}
THRES=${7}
IDRTHRES=${8}
blackListFile=${9}

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
    macs2 callpeak -g ${GenomeSize} --keep-dup all -p ${THRES} -f BAMPE -n ${PEAKN} -t ${PEAKBAM} -c ${ctrl}.sorted.uniq.nomito.bam -B
    cat ${PEAKN}_peaks.narrowPeak | cut -f 1-3 > ${PEAKN}.uncleaned.peaks.bed
    bedtools subtract -A -a ${PEAKN}.uncleaned.peaks.bed -b ${blackListFile} > ${PEAKN}.cleaned.peaks.bed
done

module unload python/2.7.12
module load python/3.6.0
module load idr/2.0.2

#### filter peaks based on IDR and filter out ENCODE blacklist regions for mm10
echo "Running idr to filter peaks..."
idr --samples ${REP1}_peaks.narrowPeak ${REP2}_peaks.narrowPeak --peak-list ${OUTP}_merge_peaks.narrowPeak --input-file-type narrowPeak --rank p.value --output-file ${OUTP}.idr --soft-idr-threshold ${IDRTHRES} --plot --log-output-file ${OUTP}.idr.log
IDRCUT=`echo "-l(${IDRTHRES})/l(10)" | bc -l`
echo "idr cut: ${IDRCUT}"
cat ${OUTP}_merge_peaks.narrowPeak | grep -w -Ff <(cat ${OUTP}.idr | awk '$12>='"${IDRCUT}"'' | cut -f 1-3) > ${OUTP}.filtered.narrowPeak

bedtools subtract -A -a ${OUTP}.filtered.narrowPeak -b ${blackListFile} > ${OUTP}.filtered.cleaned.narrowPeak 
cat ${OUTP}.filtered.cleaned.narrowPeak | cut -f 1-3 > ${OUTP}.filtered.cleaned.narrowPeak.bed
gzip ${OUTP}.idr


#### Remove bam files
echo "Remove bam files..."
rm *.bam
rm *.bai

