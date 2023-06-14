module load gcc/6.2.0  
module load bedtools/2.27.1
module load R/3.5.1
module load gsl/2.3

newNameA=${1}
newNameB=${2}
out=${3}
fileA=${4}
fileB=${5}

R CMD BATCH --quiet --no-restore --no-save count_allelic_imbalance_1.R count_allelic_imbalance_1.txt

sort -k1,1 -k2,2n ${fileA} > ${newNameA}
sort -k1,1 -k2,2n ${fileB} > ${newNameB}

bedtools intersect -a ${newNameA} -b ${newNameB} -wa -wb > ${out}.txt

R CMD BATCH --quiet --no-restore --no-save count_allelic_imbalance_2.R count_allelic_imbalance_2.txt
