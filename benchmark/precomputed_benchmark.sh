#!/usr/bin/env bash
# =============================================================================
# Benchmark WITHOUT raw reads: refine results that 3D-DNA / YaHS already produced.
#
# Mode 1 - DNA Zoo (3D-DNA):  automatic .assembly (input) + curated .review.assembly (truth) + .hic
#   DRAFT=sp.draft.assembly REVIEW=sp.review.assembly HIC=sp.hic bash benchmark/precomputed_benchmark.sh
#   (HIC_LAYOUT=<.assembly the .hic was built for>, default = DRAFT. RES=5000)
#
# Mode 2 - YaHS test data (Zenodo 7079219, ~ tens of MB, alignments already done):
#   YAHS_TEST=1 bash benchmark/precomputed_benchmark.sh
#   -> runs YaHS on LYZE01 contigs + Hi-C BED, then hicscaf on top. No truth available here, so this
#      is a smoke test / speed test; compare contact maps and scaffold stats.
#
# Mode 3 - DToL / VGP (YaHS): curated chromosome AGP as truth, pre-curation AGP as draft, .mcool/.hic of contacts
#   DRAFT=pre_curation.agp REVIEW=curated.agp HIC=sample.mcool bash benchmark/precomputed_benchmark.sh
# =============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$HERE:${PYTHONPATH:-}
OUT=${OUT:-precomputed_out}; RES=${RES:-5000}; MODEL=${MODEL:-$HERE/models/hicscaf.pt}
HICSCAF="python -m hicscaf.cli"
mkdir -p "$OUT"

if [[ -n "${YAHS_TEST:-}" ]]; then
  cd "$OUT"
  for f in LYZE01.contigs.fasta.gz LYZE01.contigs.fasta.gz.fai LYZE01.hic.bed.gz; do
    [[ -s $f ]] || wget -q https://zenodo.org/record/7079219/files/$f
  done
  [[ -s LYZE01.contigs.fa ]] || { gunzip -c LYZE01.contigs.fasta.gz > LYZE01.contigs.fa; samtools faidx LYZE01.contigs.fa; }
  [[ -s yahs_scaffolds_final.agp ]] || yahs -o yahs LYZE01.contigs.fa LYZE01.hic.bed.gz
  $HICSCAF correct --contigs LYZE01.contigs.fa --pairs LYZE01.hic.bed.gz --agp yahs_scaffolds_final.agp \
      --model "$MODEL" -o hicscaf_on_yahs
  $HICSCAF evaluate --truth yahs_scaffolds_final.agp --contigs LYZE01.contigs.fa.fai \
      YaHS=yahs_scaffolds_final.agp hicscaf=hicscaf_on_yahs.agp | tee agreement_with_yahs.md
  echo "(no reference truth for LYZE01: the table shows agreement with YaHS, not accuracy)"
  exit 0
fi

: "${DRAFT:?set DRAFT=<automatic .assembly or .agp>}"; : "${HIC:?set HIC=<.hic/.mcool>}"
EXTRA=()
[[ -n "${HIC_LAYOUT:-}" && $HIC_LAYOUT == *.assembly ]] && EXTRA+=(--layout-assembly "$HIC_LAYOUT")
[[ -n "${HIC_LAYOUT:-}" && $HIC_LAYOUT == *.agp ]] && EXTRA+=(--layout agp --layout-agp "$HIC_LAYOUT")
[[ -n "${JUICER_TOOLS:-}" ]] && EXTRA+=(--juicer-tools "$JUICER_TOOLS")
[[ -n "${CONTIGS:-}" ]] && EXTRA+=(--contigs "$CONTIGS")

$HICSCAF refine --draft "$DRAFT" --hic "$HIC" --resolution $RES --model "$MODEL" \
    --save-contacts "$OUT/contacts.npz" "${EXTRA[@]}" -o "$OUT/hicscaf"
if [[ -n "${REVIEW:-}" ]]; then
  $HICSCAF evaluate --truth "$REVIEW" --json "$OUT/results.json" \
      "automatic=$DRAFT" "hicscaf=$OUT/hicscaf.agp" | tee "$OUT/results.md"
fi
echo "Open $OUT/hicscaf.assembly in Juicebox (with the same .hic) to inspect the edits."
