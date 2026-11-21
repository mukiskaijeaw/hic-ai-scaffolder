# hic-ai-scaffolder (`hicscaf`)

**AI ช่วยจัดเรียงและกลับทิศ contig ด้วยข้อมูล Hi-C โดยต่อยอดจากผลของ YaHS / 3D-DNA**
*AI-assisted Hi-C scaffolding that corrects YaHS / 3D-DNA scaffolds (prototype).*

> สถานะ: ต้นแบบงานวิจัย (research prototype) ผลบนข้อมูลจำลองดีมาก แต่ยังต้องพิสูจน์บนข้อมูลจริง ดูหัวข้อ [ข้อจำกัด](#ข้อจำกัด--limitations)

---

## TL;DR

```bash
pip install -e .                       # numpy + torch
hicscaf train -o models/hicscaf.pt     # ฝึกบนจีโนมจำลอง ~15-20 นาทีบน CPU 4 คอร์
hicscaf correct --contigs contigs.fa --pairs hic.pairs.gz \
    --agp yahs_scaffolds_final.agp --model models/hicscaf.pt -o corrected
hicscaf evaluate --truth truth.tsv YaHS=yahs_scaffolds_final.agp AI=corrected.agp
```

---

## แนวคิด (Design)

YaHS และ 3D-DNA ใช้ heuristic: นับ contact ระหว่างปลาย contig แล้วต่อแบบ greedy/graph
ซึ่งพลาดบ่อยใน 3 กรณี: (1) contig สั้น ข้อมูลน้อย (2) บริเวณ repeat / compartment
ที่มี contact ระยะไกลหลอกตา (3) ทิศ (orientation) ของ contig ที่อยู่ติดกัน
เราจึงใช้แนว **Hybrid**: เริ่มจากผลของ YaHS แล้วให้โมเดลตรวจและแก้

```
contigs.fa + Hi-C pairs + draft AGP (YaHS / 3D-DNA)
        │
        ├─► 1. JunctionCNN   : ทุกรอยต่อใน draft → ตัด contact map 48×48 bin รอบรอยต่อ
        │                       (2 ช่อง: log-count, observed/expected) → P(รอยต่อถูก)
        │
        ├─► 2. EdgeGNN       : graph  node = contig (หรือชิ้นของ contig)
        │                       edge = คู่ contig ที่เป็น candidate (top-k contacts + รอยต่อใน draft)
        │                       feature = contact ปลาย-ปลาย 4 แบบ (HH/HT/TH/TT) × 3 สเกล (20k/100k/500k)
        │                                 + prior จาก draft + คะแนน CNN
        │                       message passing 4 ชั้น → 5 คลาส: ไม่ต่อ / HH / HT / TH / TT
        │
        └─► 3. Decoder       : รับ link จากความน่าจะเป็นมากไปน้อย, ปลายละ 1 link, ห้ามวน (union-find)
                                → corrected.agp / corrected.fa
```

* **ทำไม GNN?** การเลือกคู่ของปลาย contig เป็นปัญหาแบบ matching — edge หนึ่งจะน่าเชื่อหรือไม่ขึ้นกับว่าปลายนั้นมีคู่แข่งที่ดีกว่าหรือเปล่า
  message passing ให้ข้อมูลนี้กับโมเดลโดยตรง ซึ่ง greedy ของ YaHS ไม่มี
* **ทำไม CNN?** misjoin เห็นได้ชัดเป็นภาพ: รอยต่อที่ถูกจะมีแนวทแยงต่อเนื่อง รอยต่อผิดจะเห็นเป็น "บล็อกขาด"
* **ทำไม Hybrid?** draft จาก YaHS ถูกเกือบทั้งหมดอยู่แล้ว โมเดลจึงเรียน "แก้" ได้ง่ายกว่าเรียนจากศูนย์
  (แต่ `correct` ไม่ใส่ `--agp` ก็ scaffold จากศูนย์ได้)

### ข้อมูลฝึก: ป้ายกำกับฟรี (Free labels)

`hicscaf/simulate.py` สร้างจีโนมจำลองหลายร้อยชุด:

1. ตัดโครโมโซมเป็น contig (log-normal, เฉลี่ย 80–300 kb) สลับลำดับ และกลับทิศแบบสุ่ม
2. จำลอง Hi-C: P(contact) ∝ d^-α (α 0.8–1.4), trans 1–10%, bias ต่อ bin, A/B compartment, บริเวณ repeat ที่ map ไม่ได้
3. สร้าง **draft ที่มี error แบบ YaHS/3D-DNA**: กลับทิศผิด, สลับคู่, ย้ายตำแหน่ง, เชื่อมข้ามโครโมโซม, ต่อขาด, contig สั้นไม่ถูกวาง
4. ความลึกของข้อมูลสุ่ม 10k–80k pairs/Mb ให้โมเดลชินทั้งข้อมูลน้อยและมาก

ถ้ามี reference จริงคุณภาพสูง ใช้ `hicscaf cut-ref` + `hicscaf sim-reads` เพื่อทำแบบเดียวกันบนลำดับจริงได้

---

## ติดตั้ง (Install)

```bash
git clone https://github.com/mukiskaijeaw/hic-ai-scaffolder && cd hic-ai-scaffolder
pip install -e .            # หรือ pip install numpy torch
pip install matplotlib      # (ถ้าต้องการรูป contact map)
pytest -q tests
```

สำหรับ benchmark (YaHS, 3D-DNA, bwa, samtools, minimap2, QUAST):

```bash
conda env create -f benchmark/environment.yml && conda activate hicscaf-bench
```

## ฝึกโมเดล (Train)

```bash
hicscaf train --n-genomes 150 --cnn-epochs 8 --gnn-epochs 40 -o models/hicscaf.pt
```

ขั้นตอน: จำลองจีโนม → ฝึก JunctionCNN บน patch ของทุกรอยต่อใน draft → ใส่คะแนน CNN เป็น feature → ฝึก EdgeGNN (cross-entropy ถ่วงน้ำหนักคลาส)
มีโมเดลที่ฝึกแล้วใน `models/hicscaf.pt` (~3.7 MB) ใช้ได้ทันที

อยากให้เหมาะกับสิ่งมีชีวิตของคุณ: แก้ `simulate()` ให้ใช้ความยาวโครโมโซมจริง (`chroms={...}`),
ความยาว contig และความลึก Hi-C ที่ใกล้ข้อมูลจริง แล้วฝึกใหม่

## ใช้งาน (Run on your data)

```bash
# 1) map Hi-C เข้ากับ contig (แบบเดียวกับที่ YaHS แนะนำ)
bwa index contigs.fa
bwa mem -5SP -t 16 contigs.fa hic_R1.fq.gz hic_R2.fq.gz | samtools view -b -F 0x904 -o hic.bam -
samtools view hic.bam | python scripts/bam2pairs.py -p hic.pairs.gz -m merged_nodups.txt

# 2) YaHS (draft)
samtools faidx contigs.fa && yahs contigs.fa hic.bam -o yahs

# 3) ให้ AI แก้ผล YaHS
hicscaf correct --contigs contigs.fa --pairs hic.pairs.gz \
    --agp yahs_scaffolds_final.agp -o hicscaf_hybrid          # → .agp + .fa

# ใช้กับ 3D-DNA ได้เหมือนกัน
python scripts/assembly2agp.py contigs.FINAL.assembly > 3ddna.agp
hicscaf correct --contigs contigs.fa --pairs hic.pairs.gz --agp 3ddna.agp -o hicscaf_on_3ddna
```

`--min-prob` (ค่าเริ่ม 0.5) คุมความเข้ม: สูงขึ้น = ต่อน้อยลงแต่แม่นขึ้น

ถ้าไม่มี `bwa`/`samtools` และมี Hi-C เป็น `.bed` แบบ YaHS อยู่แล้ว ส่งไฟล์ `.bed(.gz)` ให้ `--pairs` ได้ตรงๆ

---

## ไม่มี raw reads? ใช้ผลที่ YaHS / 3D-DNA ทำเสร็จแล้ว (No raw reads needed)

raw Hi-C reads ใหญ่มาก (หลายสิบ–หลายร้อย GB) แต่ `hicscaf` ต้องการแค่ **contact matrix** + **ลำดับ contig เดิม** ซึ่งมีคนทำไว้ให้แล้ว:

| แหล่ง | ไฟล์ที่ใช้ | ใช้เป็น | ขนาดคร่าวๆ |
|---|---|---|---|
| **DNA Zoo** ([dnazoo.org/assemblies](https://www.dnazoo.org/assemblies)) | `.hic` + `.assembly` อัตโนมัติจาก 3D-DNA + `.review.assembly` ที่คน curate | draft + **เฉลย** | .hic ไม่กี่ GB |
| **YaHS test data** ([Zenodo 7079219](https://zenodo.org/record/7079219), ที่ `scripts/run_test.sh` ของ YaHS ใช้) | contigs `LYZE01.contigs.fasta.gz` + Hi-C alignment `LYZE01.hic.bed.gz` (map มาแล้ว) | smoke test YaHS → hicscaf | หลักสิบ MB |
| **Darwin Tree of Life / VGP (GenomeArk)** | AGP ก่อนและหลัง manual curation + `.mcool` / `.hic` / `.pretext` | draft + เฉลย | GB |

```bash
# DNA Zoo: ปรับ .assembly อัตโนมัติ แล้ววัดผลเทียบ .review.assembly ที่คน curate
hicscaf refine --draft species.assembly --hic species.hic --resolution 5000 -o refined
hicscaf evaluate --truth species.review.assembly automatic=species.assembly hicscaf=refined.agp
#   → refined.agp, refined.assembly (เปิดใน Juicebox ทับ .hic เดิมได้เลย)

# หรือทั้งหมดในคำสั่งเดียว
DRAFT=species.assembly REVIEW=species.review.assembly HIC=species.hic bash benchmark/precomputed_benchmark.sh

# YaHS test data (ดาวน์โหลด ~ไม่กี่สิบ MB, ไม่ต้อง map เอง)
YAHS_TEST=1 bash benchmark/precomputed_benchmark.sh

# DToL / VGP: AGP + .mcool
hicscaf refine --draft pre_curation.agp --hic sample.mcool --resolution 5000 -o refined
```

หมายเหตุ
* `.hic` อ่านด้วย `pip install hic-straw` หรือ `--juicer-tools juicer_tools.jar` (มีใน 3D-DNA: `visualize/juicebox_tools.jar`) ; `.mcool` ใช้ `pip install cooler`
* `.hic` แบบ Juicebox Assembly (JBAT, โครโมโซมชื่อ `assembly`) ใช้ `--layout assembly` (ค่าเริ่มเมื่อ draft เป็น .assembly) ;
  ถ้า `.hic` ถูกสร้างจาก .assembly อีกไฟล์ (เช่นจาก review) ให้ระบุ `--layout-assembly that.assembly`; จีโนม > 2.1 Gb ใส่ `--scale`
* `.hic` / `.mcool` ที่สร้างบน FASTA ของ scaffold ใช้ `--layout agp --layout-agp scaffolds.agp`
* `.pretext` ยังอ่านไม่ได้ (format ปิด) — ใช้ `.mcool`/`.hic` แทน
* ใช้ resolution ละเอียดที่สุดที่ไฟล์มี (แนะนำ 1 kb): ในการทดสอบ ที่ 5–10 kb โมเดลยังจับ misjoin ข้ามโครโมโซมได้ไม่ครบ (ดู RESULTS.md)

---

## วัดผลเทียบ YaHS / 3D-DNA (Evaluation & benchmark)

**ไฟล์ทดสอบเอามาจากไหน?** ต้องมี 3 อย่าง: reference ที่ดี (ใช้เป็นเฉลย), contigs, และ Hi-C reads
สคริปต์ `benchmark/run_benchmark.sh` จัดการให้ครบ 3 แบบ:

| โหมด | reference | contigs | Hi-C | ใช้เมื่อ |
|---|---|---|---|---|
| A (ค่าเริ่ม) | ยีสต์ S288C ดาวน์โหลดจาก NCBI | ตัดจาก reference (เฉลยแม่น 100%) | จำลองจาก reference | ทดสอบเร็ว ไม่ต้องหาข้อมูล |
| B | ของคุณ / NCBI | ตัดจาก reference | **ของจริงจาก SRA** (`HIC_SRR=`) | วัดกับ noise จริงของ Hi-C |
| C | ของคุณ | **assembly จริง** (HiFi/ONT) | ของจริง | สมจริงที่สุด เฉลยมาจาก minimap2 |
| offline | สุ่ม + repeat (`REF=random`) | ตัด | จำลอง | ไม่มีอินเทอร์เน็ต/ทดสอบท่อ |

```bash
bash benchmark/run_benchmark.sh                                   # โหมด A
REF_URL=<url ของ fasta.gz> HIC_SRR=SRRxxxxxxx bash benchmark/run_benchmark.sh   # โหมด B
REF=ref.fa CONTIGS=hifiasm.p_ctg.fa HIC_R1=r1.fq.gz HIC_R2=r2.fq.gz bash benchmark/run_benchmark.sh  # โหมด C
```

การเลือกข้อมูลจริง: แนะนำชุดที่มีทั้ง reference ระดับโครโมโซม, long reads และ Hi-C ของตัวเดียวกัน
เช่นชุดที่ใช้ใน benchmark ของ Hi-C scaffolders ([bioRxiv 2022.04.20.488415](https://www.biorxiv.org/content/10.1101/2022.04.20.488415.full.pdf))
หรือโครงการ Darwin Tree of Life / VGP (มี HiFi + Hi-C + assembly ที่ curate แล้ว) — ค้นใน NCBI SRA ด้วยชื่อสปีชีส์ + "Hi-C"

สคริปต์จะ: map Hi-C → รัน YaHS → รัน 3D-DNA → รัน hicscaf (hybrid บน YaHS, บน 3D-DNA, และจากศูนย์)
→ วัดผลทุกวิธีด้วยเฉลยเดียวกัน → QUAST → รูป contact map → เวลา/หน่วยความจำ (`*.time.log`)

### ตัวชี้วัด (Metrics) — `hicscaf evaluate`

| metric | ความหมาย |
|---|---|
| precision | สัดส่วนรอยต่อที่ทำ ที่ถูกทั้งคู่ที่ติดกันและทิศ |
| recall | สัดส่วนรอยต่อจริงที่หาเจอ |
| misjoin_interchrom | ต่อข้ามโครโมโซม |
| misjoin_relocation | ต่อ contig ที่จริงห่างกัน > 1 Mb |
| local_misorder | ลำดับผิดระยะใกล้ (≤ 1 Mb) |
| orientation_error | คู่ถูกแต่ทิศผิด |
| N50 / largest | ความยาว scaffold (ระวัง: N50 สูงจาก misjoin ไม่ได้แปลว่าดี) |

QUAST ให้ misassemblies, NGA50 และ genome fraction เพิ่มเติม

### ผลเบื้องต้น (Preliminary results)

ดูไฟล์ [`RESULTS.md`](RESULTS.md) — มีทั้งผลบนจีโนมจำลองที่ไม่เคยเห็นตอนฝึก และผลรัน YaHS / 3D-DNA จริงในโหมด offline

---

## โครงสร้างโค้ด (Layout)

```
hicscaf/
  io.py         FASTA, AGP, .pairs, truth table, minimap2 PAF → truth
  simulate.py   จีโนมจำลอง, Hi-C จำลอง (pairs และ FASTQ), draft ที่มี error
  features.py   graph features (end contacts หลายสเกล), contact-map patch รอบรอยต่อ
  models.py     JunctionCNN, EdgeGNN
  scaffold.py   inference + greedy decoder + baseline ที่ไม่ใช้ AI
  train.py      ฝึก CNN → GNN
  evaluate.py   ตัวชี้วัด
  precomputed.py อ่าน .assembly / .hic / .mcool (ใช้ผลสำเร็จรูปโดยไม่ต้องมี raw reads)
  cli.py        คำสั่ง hicscaf
scripts/        bam2pairs, assembly2agp (3D-DNA), plot_contact_map, sim_benchmark
benchmark/      run_benchmark.sh (จาก reads), precomputed_benchmark.sh (จาก .hic/.assembly), environment.yml
tests/          pytest
```

## ข้อจำกัด / Limitations

* โมเดลฝึกบน Hi-C **จำลอง** — ข้อมูลจริงมี pattern ที่ simulator ยังไม่มี (TAD, loop, centromere clustering, haplotig, Rabl)
  ผลบนข้อมูลจำลองจึงดีเกินจริง ต้องยืนยันด้วยโหมด B/C ก่อนสรุป
* ยังไม่ตัด contig ที่ chimeric เอง (ใช้ชิ้นที่ YaHS/3D-DNA ตัดไว้แล้ว) — ขั้นต่อไปคือให้ CNN สแกนภายใน contig ด้วย
* อ่าน `.pairs` ด้วย Python ล้วน: ข้อมูล > 100M pairs จะช้า (แปลงเป็น `.npz` ครั้งเดียวได้ด้วย `Contacts.save`)
* decoder เป็น greedy — อาจแทนด้วย ILP / beam search หรือ RL ในขั้นถัดไป
* ยังไม่รองรับ phasing / polyploid

## Roadmap

1. Fine-tune บน Hi-C จริงที่มีเฉลย (โหมด C หลายสปีชีส์) — domain adaptation จาก simulator
2. CNN สแกนภายใน contig เพื่อหา chimeric contig
3. ใส่ coverage จาก long reads / ความถี่ restriction site เป็น node feature
4. Export `.hic` / `.assembly` ให้แก้ต่อใน Juicebox

## English summary

`hicscaf` is a hybrid Hi-C scaffolder. It takes a draft scaffolding from YaHS or 3D-DNA, scores every
draft junction with a CNN on the local contact map, scores every candidate contig-end pairing with a
graph neural network (multi-scale end-contact features + draft prior + CNN score), then rebuilds
scaffolds with a best-first, cycle-free end-matching decoder. Models are trained on simulated genomes
where contigs are cut, shuffled and flipped, so labels are free. `benchmark/run_benchmark.sh` runs YaHS,
3D-DNA and hicscaf on the same contigs and Hi-C data and scores them against a reference
(join precision/recall, misjoins, orientation errors, N50, QUAST, contact maps).

License: MIT
