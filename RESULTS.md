# ผลเบื้องต้น / Preliminary results (2026-11-21)

> ⚠️ ทุกผลด้านล่างใช้ข้อมูล **จำลอง** (Hi-C จำลองด้วยโมเดล power-law เดียวกับที่ใช้ฝึก) จึงดีเกินจริงแน่นอน
> ใช้ยืนยันว่าระบบทำงานครบวงจรเท่านั้น ยังไม่ใช่หลักฐานว่าดีกว่า YaHS/3D-DNA บนข้อมูลจริง
> ขั้นต่อไปคือรันบน DNA Zoo (`.assembly` vs `.review.assembly`) ตาม README

Model: `models/hicscaf.pt` (train: 150 synthetic genomes, CPU 4 cores, ~25 min)

## 1. YaHS และ 3D-DNA ตัวจริง บนจีโนมทดสอบ offline

`REF=random bash benchmark/run_benchmark.sh` — จีโนมสุ่ม 6 โครโมโซม 13.4 Mb + repeat families,
ตัดเป็น 137 contigs (เฉลี่ย 100 kb), Hi-C จำลอง 2M read pairs → bwa mem -5SP → YaHS (commit ล่าสุด) / 3D-DNA (`-r 2`)

| method | scaffolds | N50 | joins | precision | recall | F1 | misjoins | local misorder | orientation err |
|---|---|---|---|---|---|---|---|---|---|
| YaHS | 15 | 2,426,004 | 122 | 0.951 | 0.885 | 0.917 | 0 | 6 | 0 |
| 3D-DNA | 2 | 8,723,433 | 135 | 0.963 | 0.992 | 0.977 | 4 (inter-chrom) | 1 | 0 |
| **hicscaf on YaHS** (from pairs) | 6 | 2,456,864 | 131 | **1.000** | **1.000** | **1.000** | 0 | 0 | 0 |
| **hicscaf on 3D-DNA** (from pairs) | 6 | 2,456,864 | 131 | **1.000** | **1.000** | **1.000** | 0 | 0 | 0 |
| hicscaf from scratch | 6 | 2,456,864 | 131 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| hicscaf refine from 3D-DNA `.hic` @1 kb (no reads) | 5 | 2,723,150 | 132 | 0.985–0.992 | 0.992 | 0.989–0.992 | 0–1 | 1 | 0 |
| hicscaf refine from 3D-DNA `.hic` @5 kb (no reads) | 3 | 4,048,903 | 134 | 0.970 | 0.992 | 0.981 | 3 | 1 | 0 |

* ความจริงมี 6 โครโมโซม (131 รอยต่อ) — hicscaf ได้ 6 scaffolds ตรงทุกรอยต่อ
* 3D-DNA มัก "ต่อทุกโครโมโซมเป็นก้อนเดียว" (ต้องแยกเองใน Juicebox) จึงมี misjoin ข้ามโครโมโซม
* เวลา: YaHS 6 s, hicscaf `correct` ~5 s / ~0.9 GB RAM (CPU)
* **ข้อสังเกตสำคัญ:** โหมด `refine` จาก `.hic` ทำงานดีที่ 1 kb แต่ที่ 5–10 kb CNN ยังตรวจรอยต่อผิดข้ามโครโมโซมได้ไม่ครบ
  (ลองเพิ่ม binning augmentation ตอนฝึกแล้ว ยังไม่ช่วย) → ใช้ resolution ละเอียดที่สุดที่ไฟล์มี และเป็นงานถัดไป

## 2. จีโนมจำลองที่โมเดลไม่เคยเห็น (8 genomes ต่อแถว, `scripts/sim_benchmark.py`)

draft = scaffolding ที่ใส่ error แบบ YaHS/3D-DNA (กลับทิศ, สลับ, ย้าย, ต่อข้ามโครโมโซม, ต่อขาด)

**ความลึกต่ำ 5,000 pairs/Mb**

| method | scaffolds | joins | precision | recall | F1 | misjoins | local misorder | orientation err |
|---|---|---|---|---|---|---|---|---|
| draft (YaHS-like errors) | 23 | 142 | 0.716 | 0.664 | 0.688 | 9 | 21 | 12 |
| greedy end-contact (no AI) | 64 | 100 | 0.967 | 0.683 | 0.778 | 0 | 2 | 1 |
| **hicscaf hybrid** | 9 | 156 | **0.989** | **0.977** | **0.983** | 0 | 0 | 2 |
| hicscaf from scratch | 10 | 154 | 0.985 | 0.965 | 0.975 | 0 | 1 | 2 |

**ความลึกสูง 30,000 pairs/Mb**

| method | scaffolds | joins | precision | recall | F1 | misjoins | local misorder | orientation err |
|---|---|---|---|---|---|---|---|---|
| draft (YaHS-like errors) | 24 | 140 | 0.744 | 0.679 | 0.709 | 6 | 22 | 11 |
| greedy end-contact (no AI) | 62 | 102 | 0.972 | 0.689 | 0.785 | 0 | 2 | 1 |
| **hicscaf hybrid** | 6 | 158 | **0.998** | **0.996** | **0.997** | 0 | 0 | 0 |
| hicscaf from scratch | 6 | 158 | 0.997 | 0.993 | 0.995 | 0 | 0 | 0 |

## ทำซ้ำ / Reproduce

```bash
hicscaf train --n-genomes 150 -o models/hicscaf.pt
python scripts/sim_benchmark.py --model models/hicscaf.pt --n 8 --pairs-per-mb 5000
REF=random bash benchmark/run_benchmark.sh          # ต้องมี bwa samtools yahs gawk java (ดู environment.yml)
```
