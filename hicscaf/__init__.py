"""hicscaf: AI-assisted Hi-C scaffolding that corrects YaHS / 3D-DNA drafts.

Pipeline (hybrid):
  1. JunctionCNN scores every junction in the draft scaffolds (is this join real?).
  2. EdgeGNN scores every candidate contig-end pairing (adjacency + orientation),
     using Hi-C end-contact features, the draft adjacency prior and the CNN score.
  3. A greedy end-matching decoder rebuilds the scaffolds.
"""
__version__ = "0.1.0"
