#### Load modules
module load gcc/6.2.0  
module load bcftools/1.9

fafile=${1}
bamfile=${2}
vcfname=${3}


bcftools mpileup -a AD,DP,SP -O z -o mpileup_${vcfname}.vcf.gz -f ${fafile} ${bamfile}