"""Command line: hicscaf {simulate,train,correct,refine,evaluate,studio,sim-reads,cut-ref,paf-truth}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_simulate(a):
    from .io import write_agp, write_pairs, write_truth
    from .simulate import simulate

    sim = simulate(seed=a.seed, pairs_per_mb=a.pairs_per_mb)
    p = Path(a.out); p.mkdir(parents=True, exist_ok=True)
    with open(p / "contigs.sizes", "w") as fh:
        fh.writelines(f"{k}\t{v}\n" for k, v in sim.lengths.items())
    sim.contacts.save(p / "contacts.npz")
    if a.write_pairs:
        write_pairs(sim.contacts, p / "contacts.pairs.gz")
    write_truth(sim.truth, p / "truth.tsv")
    write_agp(sim.draft, p / "draft.agp")
    write_agp(sim.true_scaffolds, p / "true.agp")
    print(f"wrote {p}: {len(sim.lengths)} contigs, {len(sim.contacts):,} contacts, {len(sim.draft)} draft scaffolds")


def cmd_train(a):
    from .train import main
    main(a)


def cmd_correct(a):
    from .io import Contacts, agp_to_fasta, read_agp, read_fasta, read_lengths, write_agp, write_fasta
    from .models import load
    from .scaffold import correct

    lengths = read_lengths(a.contigs)
    contacts = Contacts.load(a.pairs)
    from .precomputed import load_scaffolds
    draft = load_scaffolds(a.agp)[0] if a.agp else None
    cnn, gnn, _ = load(a.model)
    from .hicwrite import can_write_hic
    hic_out = a.out + ".hic" if (not a.no_hic and can_write_hic()) else None
    scafs = correct(lengths, contacts, draft, cnn, gnn, min_prob=a.min_prob, bin_size=a.bin_size,
                    n_chrom=a.n_chrom, hic_out=hic_out)
    write_agp(scafs, a.out + ".agp")
    from .precomputed import write_assembly
    write_assembly(scafs, lengths, a.out + ".assembly")
    if not str(a.contigs).endswith((".fai", ".sizes", ".tsv")):
        write_fasta(agp_to_fasta(scafs, read_fasta(a.contigs)), a.out + ".fa")
    print(f"wrote {a.out}.agp ({len(scafs)} scaffolds)" + (f" and {hic_out} (open in Juicebox with {a.out}.assembly)" if hic_out else ""))


def cmd_refine(a):
    """Refine an existing YaHS / 3D-DNA result using its contact matrix only (no raw reads)."""
    from .io import agp_to_fasta, read_fasta, read_lengths, write_agp, write_fasta
    from .models import load
    from .precomputed import (layout_from_agp, layout_from_assembly, load_matrix_contacts, load_scaffolds,
                              write_assembly)
    from .scaffold import correct

    draft, lengths = load_scaffolds(a.draft)
    if a.contigs:
        lengths = read_lengths(a.contigs)
    names = list(lengths)
    layout_kind = a.layout or ("assembly" if a.draft.endswith(".assembly") else "agp")
    if layout_kind == "assembly":
        layout = layout_from_assembly(load_scaffolds(a.layout_assembly)[0] if a.layout_assembly else draft, names)
    else:
        layout = layout_from_agp(a.layout_agp or a.draft, names)
    contacts = load_matrix_contacts(a.hic, layout, a.resolution, a.juicer_tools, a.scale)
    print(f"loaded {contacts.w.sum():,.0f} contacts ({len(contacts):,} non-zero bins at {a.resolution} bp) "
          f"for {len(draft)} draft scaffolds")
    if a.save_contacts:
        contacts.save(a.save_contacts)
    cnn, gnn, _ = load(a.model)
    from .hicwrite import can_write_hic
    hic_out = a.out + ".hic" if (not a.no_hic and can_write_hic()) else None
    scafs = correct(lengths, contacts, draft, cnn, gnn, min_prob=a.min_prob, bin_size=a.bin_size,
                    n_chrom=a.n_chrom, hic_out=hic_out, hic_resolution=a.resolution * a.scale)
    write_agp(scafs, a.out + ".agp")
    write_assembly(scafs, lengths, a.out + ".assembly")  # open in Juicebox (JBAT) for manual review
    if a.contigs and not a.contigs.endswith((".fai", ".sizes", ".tsv")):
        write_fasta(agp_to_fasta(scafs, read_fasta(a.contigs)), a.out + ".fa")
    print(f"wrote {a.out}.agp ({len(scafs)} scaffolds)" + (f" and {hic_out} (open in Juicebox with {a.out}.assembly)" if hic_out else ""))


def _load_truth(path):
    from .io import read_truth, truth_from_scaffolds
    if path.endswith((".assembly", ".agp")):
        from .precomputed import load_scaffolds
        return truth_from_scaffolds(load_scaffolds(path)[0])
    return read_truth(path)


def cmd_evaluate(a):
    from .evaluate import evaluate, format_table
    from .io import read_lengths, add_missing_components
    from .precomputed import load_scaffolds

    truth = _load_truth(a.truth)
    lengths = read_lengths(a.contigs) if a.contigs else None
    rows = {}
    for spec in a.agp:
        name, _, path = spec.rpartition("=")
        scafs, _ = load_scaffolds(path)
        if lengths:
            scafs = add_missing_components(scafs, lengths)
        rows[name or Path(path).stem] = evaluate(scafs, truth)
    table = format_table(rows)
    print(table)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))


def cmd_sim_reads(a):
    from .io import read_fasta
    from .simulate import simulate_hic_reads
    simulate_hic_reads(read_fasta(a.ref), a.n_pairs, a.out, read_len=a.read_len, seed=a.seed)


def cmd_cut_ref(a):
    from .io import read_fasta, write_fasta, write_truth
    from .simulate import cut_reference_into_contigs
    seqs, truth = cut_reference_into_contigs(read_fasta(a.ref), seed=a.seed, mean_len=a.mean_len)
    write_fasta(seqs, a.out + ".fa")
    write_truth(truth, a.out + ".truth.tsv")
    print(f"wrote {len(seqs)} contigs to {a.out}.fa")


def cmd_random_ref(a):
    """Random genome with interspersed repeat families (for offline tests when no reference can be downloaded)."""
    import numpy as np
    from .io import write_fasta
    rng = np.random.default_rng(a.seed)
    alphabet = np.frombuffer(b"ACGT", np.uint8)
    fams = [alphabet[rng.integers(0, 4, int(rng.integers(2_000, 12_000)))] for _ in range(a.repeat_families)]
    seqs = {}
    for i in range(a.n_chrom):
        L = int(rng.integers(a.min_len, a.max_len))
        s = alphabet[rng.integers(0, 4, L)]
        for _ in range(int(L / 1e6 * a.repeats_per_mb)):
            f = fams[int(rng.integers(len(fams)))].copy()
            mut = rng.random(len(f)) < 0.01
            f[mut] = alphabet[rng.integers(0, 4, mut.sum())]
            p = int(rng.integers(0, L - len(f)))
            s[p : p + len(f)] = f
        seqs[f"chr{i + 1}"] = s.tobytes().decode()
    write_fasta(seqs, a.out)
    print(f"wrote {a.out}: {len(seqs)} chromosomes, {sum(map(len, seqs.values())):,} bp")


def cmd_studio(a):
    from . import studio
    if a.public:
        studio.PUBLIC = True
    studio.serve(a.host, a.port, a.workdir, not a.no_browser)


def cmd_paf_truth(a):
    from .io import truth_from_paf, write_truth
    write_truth(truth_from_paf(a.paf), a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hicscaf", description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("simulate", help="write one synthetic example (contigs, contacts, draft, truth)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--pairs-per-mb", type=int, default=40_000)
    p.add_argument("--write-pairs", action="store_true"); p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_simulate)

    p = sp.add_parser("train", help="train CNN + GNN on synthetic genomes")
    p.add_argument("--n-genomes", type=int, default=120); p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--gnn-epochs", type=int, default=40); p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4); p.add_argument("--seed", type=int, default=1)
    p.add_argument("-o", "--out", default="models/hicscaf.pt")
    p.set_defaults(fn=cmd_train)

    p = sp.add_parser("correct", help="correct a YaHS/3D-DNA AGP (or scaffold from scratch without --agp)")
    p.add_argument("--contigs", required=True, help="contigs FASTA (or .fai / .sizes)")
    p.add_argument("--pairs", required=True, help=".pairs[.gz] / 4-col TSV / .npz")
    p.add_argument("--agp", help="draft AGP from YaHS or 3D-DNA (omit = scaffold from scratch)")
    p.add_argument("--model", default="models/hicscaf.pt"); p.add_argument("--min-prob", type=float, default=0.5)
    p.add_argument("--bin-size", type=int); p.add_argument("-o", "--out", required=True)
    p.add_argument("--n-chrom", type=int, default=0,
                   help="expected number of chromosomes (haploid n): group scaffolds into this many chromosomes")
    p.add_argument("--no-hic", action="store_true",
                   help="do not write <out>.hic (Juicebox assembly-mode map of the result; open with <out>.assembly)")
    p.set_defaults(fn=cmd_correct)

    p = sp.add_parser("refine", help="refine a finished YaHS/3D-DNA result from its .hic/.mcool (no raw reads)")
    p.add_argument("--draft", required=True, help=".assembly (3D-DNA/DNA Zoo) or .agp (YaHS/DToL/VGP)")
    p.add_argument("--hic", required=True, help=".hic, .mcool/.cool, or text dump")
    p.add_argument("--layout", choices=["assembly", "agp"],
                   help="coordinate system of the matrix (default: assembly for .assembly, agp for .agp)")
    p.add_argument("--layout-agp", help="AGP that describes the matrix coordinates if different from --draft")
    p.add_argument("--layout-assembly",
                   help=".assembly the .hic was built from, if different from --draft (e.g. a .hic made for the review)")
    p.add_argument("--resolution", type=int, default=5000)
    p.add_argument("--scale", type=int, default=1, help="Juicebox scale factor (genomes > 2.1 Gb)")
    p.add_argument("--juicer-tools", help="juicer_tools.jar, used when hic-straw is not installed")
    p.add_argument("--contigs", help="contigs FASTA (optional; for lengths and FASTA output)")
    p.add_argument("--model", default="models/hicscaf.pt"); p.add_argument("--min-prob", type=float, default=0.5)
    p.add_argument("--bin-size", type=int); p.add_argument("--save-contacts", help="cache contacts as .npz")
    p.add_argument("-o", "--out", required=True, help="output prefix (.agp, .assembly, .fa)")
    p.add_argument("--n-chrom", type=int, default=0,
                   help="expected number of chromosomes (haploid n): group scaffolds into this many chromosomes")
    p.add_argument("--no-hic", action="store_true",
                   help="do not write <out>.hic (Juicebox assembly-mode map of the result; open with <out>.assembly)")
    p.set_defaults(fn=cmd_refine)

    p = sp.add_parser("evaluate", help="compare scaffoldings (.agp/.assembly) against a truth")
    p.add_argument("--truth", required=True,
                   help="truth.tsv, or a curated .review.assembly / .agp used as ground truth"); p.add_argument("--contigs", help="FASTA/.fai/.sizes (adds unplaced contigs)")
    p.add_argument("--json"); p.add_argument("agp", nargs="+", help="name=path.agp or name=path.assembly")
    p.set_defaults(fn=cmd_evaluate)

    p = sp.add_parser("sim-reads", help="simulate Hi-C FASTQ from a reference")
    p.add_argument("--ref", required=True); p.add_argument("--n-pairs", type=int, default=2_000_000)
    p.add_argument("--read-len", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_sim_reads)

    p = sp.add_parser("cut-ref", help="cut a reference into shuffled/flipped contigs + truth table")
    p.add_argument("--ref", required=True); p.add_argument("--mean-len", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=0); p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_cut_ref)

    p = sp.add_parser("random-ref", help="random reference with repeats (offline testing)")
    p.add_argument("--n-chrom", type=int, default=6); p.add_argument("--min-len", type=int, default=1_000_000)
    p.add_argument("--max-len", type=int, default=3_000_000); p.add_argument("--repeat-families", type=int, default=20)
    p.add_argument("--repeats-per-mb", type=float, default=15); p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_random_ref)

    p = sp.add_parser("paf-truth", help="truth table from minimap2 PAF (contigs vs reference)")
    p.add_argument("paf"); p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_paf_truth)

    p = sp.add_parser("studio", help="open the hicscaf Studio web app (upload files, run, view results)")
    import os
    p.add_argument("--host", default=os.environ.get("HICSCAF_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    p.add_argument("--public", action="store_true",
                   help="server mode for other people: uploads only (no server paths), size/job limits; "
                        "set HICSCAF_PASSWORD to require a password")
    p.add_argument("--workdir", help="where uploads and results are kept (default ~/hicscaf_studio)")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_studio)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
