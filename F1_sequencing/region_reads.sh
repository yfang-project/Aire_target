#### Load modules
module load gcc/6.2.0  
module load bcftools/1.9


vcffile1=${1}
vcffile2=${2}
vcfDir1=${3}
vcfDir2=${4}
vcfname=${5}
region=${6}

cp ${vcfDir1} .
cp ${vcfDir2} .

bcftools view -R ${region} -O z -o ${vcfname}.vcf.gz ${vcffile1}

rm ${vcffile1} ${vcffile2}

bcftools view -f PASS -O z -o ${vcfname}.filtered.vcf.gz ${vcfname}.vcf.gz
bcftools query -f '%CHROM\t%POS\n' -o ${vcfname}.filtered.region.txt ${vcfname}.filtered.vcf.gz
sed -e 's/^/chr/' ${vcfname}.filtered.region.txt > ${vcfname}_have_chr.filtered.region.txt