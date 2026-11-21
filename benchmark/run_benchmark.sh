#!/usr/bin/env bash
# =============================================================================
# hicscaf benchmark: YaHS vs 3D-DNA vs hicscaf (hybrid on YaHS, and from scratch)
#
# Truth comes from a reference genome:
#   * contigs  : either cut from the reference (exact truth, default) or your own
#                assembly (truth = minimap2 placement of contigs on the reference)
#   * Hi-C     : either simulated from the reference (default, no download) or
#                real reads from SRA (set HIC_SRR) / your own FASTQ (HIC_R1/HIC_R2)
#
# usage:
#   bash benchmark/run_benchmark.sh                        # yeast, everything simulated
#   REF_URL=<fasta.gz url> HIC_SRR=SRRxxxxxxx bash benchmark/run_benchmark.sh
#   REF=random bash benchmark/run_benchmark.sh            # fully offline smoke test
#   REF=ref.fa CONTIGS=my_asm.fa HIC_R1=r1.fq.gz HIC_R2=r2.fq.gz bash benchmark/run_benchmark.sh
# =============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-bench_out}
THREADS=${THREADS:-8}
MODEL=${MODEL:-$HERE/models/hicscaf.pt}
REF_URL=${REF_URL:-https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz}
N_PAIRS=${N_PAIRS:-2000000}       # simulated Hi-C read pairs
MEAN_CONTIG=${MEAN_CONTIG:-100000}
THREEDDNA=${THREEDDNA:-$OUT/3d-dna}
HICSCAF="python -m hicscaf.cli"
export PYTHONPATH=$HERE:${PYTHONPATH:-}

mkdir -p "$OUT"; cd "$OUT"
log(){ echo -e "\n[$(date +%T)] $*"; }

# ---------------------------------------------------------------- 1. reference
if [[ "${REF:-}" == "random" ]]; then
  REF=ref.fa
  [[ -s $REF ]] || { log "no download: generate a random reference with repeats"; $HICSCAF random-ref -o $REF; }
elif [[ -z "${REF:-}" ]]; then
  REF=ref.fa
  [[ -s $REF ]] || { log "download reference"; wget -qO- "$REF_URL" | gunzip -c > $REF; }
fi

# ---------------------------------------------------------------- 2. contigs + truth
if [[ -z "${CONTIGS:-}" ]]; then
  log "cut reference into shuffled/flipped contigs (mean ${MEAN_CONTIG} bp)"
  $HICSCAF cut-ref --ref "$REF" --mean-len $MEAN_CONTIG -o contigs
  CONTIGS=contigs.fa; TRUTH=contigs.truth.tsv
else
  log "place your contigs on the reference with minimap2 -> truth"
  minimap2 -x asm5 -t $THREADS "$REF" "$CONTIGS" > contigs_vs_ref.paf
  $HICSCAF paf-truth contigs_vs_ref.paf -o contigs.truth.tsv; TRUTH=contigs.truth.tsv
fi
samtools faidx "$CONTIGS"

# ---------------------------------------------------------------- 3. Hi-C reads
if [[ -n "${HIC_SRR:-}" ]]; then
  log "download Hi-C $HIC_SRR from SRA"
  [[ -s ${HIC_SRR}_1.fastq.gz ]] || { fasterq-dump --split-files -e $THREADS "$HIC_SRR"; pigz -p $THREADS ${HIC_SRR}_[12].fastq; }
  HIC_R1=${HIC_SRR}_1.fastq.gz; HIC_R2=${HIC_SRR}_2.fastq.gz
elif [[ -z "${HIC_R1:-}" ]]; then
  log "simulate $N_PAIRS Hi-C read pairs from the reference"
  [[ -s hic_R1.fastq.gz ]] || $HICSCAF sim-reads --ref "$REF" --n-pairs $N_PAIRS -o hic
  HIC_R1=hic_R1.fastq.gz; HIC_R2=hic_R2.fastq.gz
fi

# ---------------------------------------------------------------- 4. map Hi-C to contigs
if [[ ! -s hic.bam ]]; then
  log "bwa mem -5SP (Arima/YaHS recommended settings)"
  bwa index "$CONTIGS"
  bwa mem -5SP -t $THREADS "$CONTIGS" "$HIC_R1" "$HIC_R2" | samtools view -b -F 0x904 -o hic.bam -
fi
[[ -s hic.pairs.gz ]] || samtools view hic.bam | python "$HERE/scripts/bam2pairs.py" --mapq 10 -p hic.pairs.gz -m merged_nodups.txt

# ---------------------------------------------------------------- 5. YaHS
log "YaHS"
[[ -s yahs_scaffolds_final.agp ]] || ( /usr/bin/time -v yahs "$CONTIGS" hic.bam -o yahs ) 2> yahs.time.log

# ---------------------------------------------------------------- 6. 3D-DNA
log "3D-DNA"
if [[ ! -d $THREEDDNA ]]; then git clone --depth 1 https://github.com/aidenlab/3d-dna "$THREEDDNA"; fi
# 3D-DNA hard-codes a 48 GB Java heap; small genomes need far less
sed -i "s/-Xms49152m -Xmx49152m/-Xmx${JAVA_MEM:-8g}/" "$THREEDDNA/visualize/juicebox_tools.sh"
if [[ ! -s 3ddna.agp ]]; then
  mkdir -p 3ddna_run
  ( cd 3ddna_run && /usr/bin/time -v bash "$THREEDDNA/run-asm-pipeline.sh" -r 2 "../$CONTIGS" ../merged_nodups.txt ) \
      > 3ddna.log 2> 3ddna.time.log || echo "3D-DNA failed - see 3ddna.log"
  FINAL=$(ls 3ddna_run/*.FINAL.assembly 2>/dev/null | head -1 || true)
  [[ -n $FINAL ]] && python "$HERE/scripts/assembly2agp.py" "$FINAL" > 3ddna.agp
fi

# ---------------------------------------------------------------- 7. hicscaf
log "hicscaf hybrid (correct YaHS) + from scratch"
( /usr/bin/time -v $HICSCAF correct --contigs "$CONTIGS" --pairs hic.pairs.gz --agp yahs_scaffolds_final.agp \
     --model "$MODEL" -o hicscaf_hybrid ) 2> hicscaf_hybrid.time.log
( /usr/bin/time -v $HICSCAF correct --contigs "$CONTIGS" --pairs hic.pairs.gz --model "$MODEL" -o hicscaf_scratch ) \
     2> hicscaf_scratch.time.log
[[ -s 3ddna.agp ]] && $HICSCAF correct --contigs "$CONTIGS" --pairs hic.pairs.gz --agp 3ddna.agp --model "$MODEL" -o hicscaf_on_3ddna

# ---------------------------------------------------------------- 8. evaluate
log "join-level evaluation against the reference"
AGPS="YaHS=yahs_scaffolds_final.agp hicscaf_hybrid=hicscaf_hybrid.agp hicscaf_scratch=hicscaf_scratch.agp"
[[ -s 3ddna.agp ]] && AGPS="$AGPS 3D-DNA=3ddna.agp hicscaf_on_3D-DNA=hicscaf_on_3ddna.agp"
$HICSCAF evaluate --truth "$TRUTH" --contigs "$CONTIGS.fai" --json results.json $AGPS | tee results.md

log "QUAST (reference-based misassemblies, NGA50)"
FAS="yahs_scaffolds_final.fa hicscaf_hybrid.fa hicscaf_scratch.fa"
[[ -d 3ddna_run ]] && FAS="$FAS $(ls 3ddna_run/*.FINAL.fasta 2>/dev/null || true)"
quast.py -r "$REF" -t $THREADS --large --scaffold-gap-max-size 1000 --min-contig 1000 -o quast $FAS > quast.log 2>&1 \
   && cat quast/report.txt || echo "QUAST failed - see quast.log"

log "contact maps"
for x in $AGPS; do
  n=${x%%=*}; f=${x#*=}
  python "$HERE/scripts/plot_contact_map.py" --contigs "$CONTIGS.fai" --pairs hic.pairs.gz --agp "$f" --title "$n" -o "map_$n.png" || true
done
grep -H "Elapsed\|Maximum resident" *.time.log || true
log "done: see $OUT/results.md, $OUT/quast/report.txt, $OUT/map_*.png"
