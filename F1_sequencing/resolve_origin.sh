module load gcc/6.2.0
module load python/2.7.12

## Ref
modfile1=${1}
bamfile1=${2}
bamgzfile1=${3}
baifile1=${4}
modfiledir1=${5}

## Alt
modfile2=${6}
bamfile2=${7}
bamgzfile2=${8}
baifile2=${9}
modfiledir2=${10}

mergebam=${11}

cp ${bamgzfile1} .
cp ${baifile1} .
cp ${modfiledir1} .

cp ${bamgzfile2} .
cp ${baifile2} .
cp ${modfiledir2} .


gunzip -fv *.gz

pylapels -n -p 8 ${modfile1} ${bamfile1}
pylapels -n -p 8 ${modfile2} ${bamfile2}

pysuspenders -t ./${mergebam} ./${bamfile1} ./${bamfile2}


