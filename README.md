# Genome skimming

Snakemake pipeline for for genome skimming from fastq or bam files. The pipeline generates a matrix with pairwise distances between samples based on kmers.

## 1. Prepare environment

Clone this repository and rename the main directory as wished. Replace [project_name] by an actual project name

```
screen -S [project_name]
git clone https://github.com/alberdilab/genome_skimming
mv genome_skimming [project_name]
cd [project_name]
```

Download Snakemake wrappers to avoid connection issues before launching the pipeline.

```
git clone --depth 1 --branch v7.2.0 https://github.com/snakemake/snakemake-wrappers.git  workflow/wrappers/v7.2.0
```

## 2. Prepare input

Prepare sequencing reads or mapped bam files.

- **reads**: add sequencing reads to the **reads** directory.
- **bams**: add bam files to the **mapped** directory.

## 3. Launch pipeline

```
snakemake --workflow-profile profile/slurm
```

## 4. View results

The final output is a single file called **distance_matrix.txt**.

```
head distance_matrix.txt
```
