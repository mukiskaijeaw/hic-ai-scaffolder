# HANDOFF: hic-ai-scaffolder (สถานะ ณ 2026-11-21)

ไฟล์นี้สรุปทุกอย่างที่ต้องรู้เพื่อทำงานต่อ (ใช้กับ ChatGPT / Codex หรือคนอื่นก็ได้)
Repo: https://github.com/mukiskaijeaw/hic-ai-scaffolder (branch หลัก `main`)

---

## 1. โปรเจกต์นี้คืออะไร

`hicscaf` = เครื่องมือ Hi-C scaffolding ที่ใช้ AI ไปแก้ผลของ YaHS / 3D-DNA

- **JunctionCNN**: ดู contact map รอบรอยต่อแต่ละจุดใน draft แล้วบอกว่ารอยต่อถูกหรือผิด
- **EdgeGNN**: ทำนายว่าปลาย contig คู่ไหนควรต่อกัน (5 คลาส: ไม่ต่อ, HH, HT, TH, TT)
- **Greedy decoder**: ต่อจากคู่ที่มั่นใจที่สุดลงไป ไม่ให้เกิดวง (union-find)
- **Grouping** (`--n-chrom N`): จัดเป็น N โครโมโซม แล้วเรียงและกลับทิศในแต่ละกลุ่ม
- ฝึกด้วยจีโนมจำลอง โมเดลที่ฝึกแล้วอยู่ที่ `models/hicscaf.pt`

Python package: `hicscaf/`

| ไฟล์ | หน้าที่ |
|---|---|
| `cli.py` | คำสั่ง `hicscaf ...` |
| `scaffold.py` | inference + decoder |
| `grouping.py` | จัดกลุ่มโครโมโซม (`--n-chrom`) |
| `features.py` | graph features, CNN patches |
| `models.py` | CNN + GNN |
| `precomputed.py` | อ่าน .assembly / .agp / .hic / .mcool |
| `hicfile.py` | ตัวอ่าน .hic แบบ pure Python (สำรอง) |
| `hicwrite.py` | เขียน .hic ของผลลัพธ์ (Juicebox assembly mode) |
| `studio.py` + `studio_web/` | หน้าเว็บ hicscaf Studio (http.server ของ stdlib) |
| `simulate.py` | จีโนมจำลอง |
| `evaluate.py` | ตัวชี้วัด |

Tests อยู่ใน `tests/` (`python -m pytest -q` ต้องผ่าน 19 ข้อ)

## 2. ติดตั้งบน Windows (เครื่อง muk: `D:\hic\hic-ai-scaffolder`, มี `.venv` อยู่แล้ว)

PowerShell บล็อก `Activate.ps1` ให้เรียก python ของ venv ตรง ๆ แทน:

```bat
cd /d D:\hic\hic-ai-scaffolder
git checkout main
git pull
.venv\Scripts\python.exe -m pip install -e .
```

Dependencies (อยู่ใน `pyproject.toml`): numpy, torch, hictkpy, scipy, pandas, pyarrow
(hic-straw build บน Windows ไม่ได้ จึงไม่ใช้)

เปิดหน้าเว็บ: ดับเบิลคลิก `studio.bat` หรือรัน `.venv\Scripts\python.exe -m hicscaf.cli studio`
แล้วเปิด http://127.0.0.1:8765

## 3. คำสั่งที่ใช้บ่อย

```bat
:: แก้ draft จาก .hic + .assembly แล้วจัดเป็น 7 โครโมโซม (ได้ out.agp, out.assembly, out.hic)
.venv\Scripts\python.exe -m hicscaf.cli refine --draft ASM360417v1.assembly --hic ASM360417v1.hic ^
    --resolution 5000 --n-chrom 7 -o out

:: เปิดผลใน Juicebox 2.x: File > Open > out.hic แล้ว Assembly > Import Map Assembly > out.assembly

:: วัดผลเทียบกับคำตอบจริง (ต้องมี truth.tsv จาก minimap2 ก่อน ดูข้อ 5)
.venv\Scripts\python.exe -m hicscaf.cli evaluate --truth truth.tsv --contigs ASM360417v1.fa.fai hicscaf=out.agp

:: ทดสอบ
.venv\Scripts\python.exe -m pytest -q
```

ตัวเลือกที่สำคัญ:
- `--min-prob` (0.5): สูงขึ้น = ต่อน้อยลงแต่แม่นขึ้น
- `--n-chrom` (0 = ปิด)
- `--no-hic`: ไม่เขียน .hic
- `--resolution`: ใช้ค่าที่ละเอียดที่สุดที่ไฟล์ .hic มี

## 4. ข้อมูลของ muk

- สิ่งมีชีวิต: พยาธิใบไม้ตับ **Clonorchis sinensis**, haploid n = **7**
- Draft: `ASM360417v1.assembly` (WGS NIRI01)
  - 2,776 contig, 562.8 Mb
  - ยังไม่ได้ทำ scaffold
- Hi-C: `ASM360417v1.hic`
  - เป็น .hic v8 มี pseudo-chromosome เดียวชื่อ "assembly"
  - มี contact แค่ 1.49 ล้านคู่ (บางมาก ประมาณ 2,600 ต่อ Mb)
  - ครึ่งหนึ่งของ contact อยู่ภายใน contig เดียวกัน
- คำตอบจริง: **CSKR.v2 = GCA_003604175.2** มี 7 โครโมโซม ขนาด 222 / 169 / 37 / 33 / 32 / 31 / 30 Mb

## 5. ปัญหาที่ค้างอยู่: โครโมโซมใหญ่ถูกแบ่ง

ผลของ `--n-chrom 7` บนข้อมูลจริงได้ 7 กลุ่ม ขนาด 173 / 102 / 97 / 73 / 55 / 30 / 30 Mb
คำตอบจริงคือ 222 / 169 / 37 / 33 / 32 / 31 / 30 Mb
แปลว่า 2 โครโมโซมใหญ่ถูกแบ่งเป็นหลายกลุ่ม แล้วไปปนกับโครโมโซมเล็ก มีแค่ 2 กลุ่มสุดท้ายที่ตรง

สิ่งที่รู้แล้ว:
- **จีโนมจำลองไม่เจอปัญหานี้**
  - สร้างจีโนมขนาด 222/169/37/33/32/31/30 Mb, contig เฉลี่ย 200 kb, 2,600 pairs/Mb
  - trans contact 3% และ 30% ให้ผลตรง 100% ทั้งคู่
  - แปลว่าปัญหามาจากลักษณะเฉพาะของข้อมูลจริง ไม่ได้มาจากขนาดโครโมโซม
- **ไม่ได้เกิดจากการต่อของ AI**: ลองจัดกลุ่มจาก contig เดี่ยว ๆ ก็ได้ขนาดผิดแบบเดียวกัน
- **ลองวิธี modularity แล้ว** (`method="modularity"` ใน `grouping._cluster`) ได้ 150/116/98/66/47/43/41 ก็ยังไม่ตรง
  ตอนนี้ค่าเริ่มต้นยังเป็น `density` (average linkage แบบ LACHESIS)
- **contact ข้ามกลุ่มสูงและค่อนข้างเท่ากันทุกคู่**
  - ความหนาแน่นนอกแนวทแยงประมาณ 2–4 ต่อ Mb²
  - ความหนาแน่นในกลุ่มลดลงตามขนาด: 10 ที่ 173 Mb เทียบกับ 33 ที่ 30 Mb
  - average linkage ตามความหนาแน่นจึงไม่อยากรวมโครโมโซมใหญ่

ขั้นต่อไปที่แนะนำ:
1. ทำคำตอบจริงระดับ contig บน PC (ต้องโหลด CSKR.v2 ประมาณ 170 MB และ minimap2):
   ```
   minimap2 -x asm5 -t 8 GCA_003604175.2.fna ASM360417v1_contigs.fa > ctg_vs_ref.paf
   hicscaf paf-truth ctg_vs_ref.paf -o truth.tsv
   ```
2. ใช้ truth.tsv หาว่ากลุ่มไหนปนกับโครโมโซมไหน แล้วเช็ก 3 อย่าง:
   - contig ที่ผิดเป็นตัวเล็กหรือตัวใหญ่
   - มี contig chimeric (contig เดียวที่คร่อม 2 โครโมโซม) หรือไม่
   - มี contig ที่ coverage สูงผิดปกติ (repeat) ซึ่งดึงกลุ่มให้ติดกันหรือไม่
3. ไอเดียที่ควรลอง:
   - normalize contact แบบ ICE / coverage (หาร T_i·T_j) ก่อนจัดกลุ่ม
   - ตัด contig ที่ coverage สูงผิดปกติออกก่อนจัดกลุ่ม (แบบที่ ALLHiC / LACHESIS ทำ)
   - ใช้ spectral clustering บนเมทริกซ์ obs/exp
   - ใช้ single-linkage บนเฉพาะ link ที่แข็งแรง เพื่อไล่ตามสายของโครโมโซม
   - ตัด contig ที่ chimeric ก่อน
4. วัดผลด้วย `hicscaf evaluate` ทั้งกับจีโนมจำลองและกับ truth.tsv
   ผลต้องไม่แย่ลงบนจีโนมจำลอง (`tests/test_grouping.py`)

## 6. ไฟล์ผลลัพธ์ที่ทำไว้แล้ว

- ไฟล์ชุดล่าสุดใช้ `--n-chrom 7` และวิธี density:
  `ASM360417v1.7chrom_5kb.{assembly,agp,hic}`
- ไฟล์ชุดก่อนหน้ายังไม่ได้จัดกลุ่ม:
  `ASM360417v1.corrected_1kb.{assembly,agp}`

## 7. Prompt สำหรับเริ่มใน Codex / ChatGPT

> ผมทำงานบน repo github.com/mukiskaijeaw/hic-ai-scaffolder (Python package `hicscaf`)
> อ่าน `HANDOFF.md` และ `README.md` ก่อน แล้วช่วยแก้ปัญหาในข้อ 5 (`hicscaf/grouping.py`)
> ให้โครโมโซมใหญ่ไม่ถูกแบ่ง ห้ามทำให้ `python -m pytest -q` พัง
> ผมใช้ Windows เรียก python ผ่าน `.venv\Scripts\python.exe`
