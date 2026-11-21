(function(){
"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const bp=x=>x>=1e9?(x/1e9).toFixed(2)+" Gb":x>=1e6?(x/1e6).toFixed(2)+" Mb":x>=1e3?(x/1e3).toFixed(1)+" kb":x+" bp";
const human=b=>b>1e9?(b/1e9).toFixed(1)+" GB":b>1e6?(b/1e6).toFixed(0)+" MB":(b/1e3).toFixed(0)+" KB";
const num=x=>Number(x).toLocaleString("en-US");
const dur=t=>{t=Math.max(0,Math.round(t||0));if(t<60)return t+" วินาที";const m=Math.floor(t/60);if(m<60)return m+" นาที"+(m<10&&t%60?" "+(t%60)+" วินาที":"");const h=Math.floor(m/60);return h+" ชม. "+(m%60)+" นาที"};
async function api(path,opt){const r=await fetch(path,opt);let j;try{j=await r.json()}catch(e){throw new Error("เซิร์ฟเวอร์ไม่ตอบ (หน้าต่างโปรแกรมยังเปิดอยู่ไหม)")}if(!r.ok)throw new Error(j.error||r.statusText);return j}
const post=(p,b)=>api(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});

const SLOTS={
  correct:[
    {k:"contigs",t:"Contigs",ext:".fa / .fasta / .fai",acc:".fa,.fasta,.fna,.gz,.fai,.sizes",req:true,ico:".fa"},
    {k:"pairs",t:"Hi-C pairs",ext:".bed.gz / .pairs.gz / .npz",acc:".bed,.gz,.pairs,.npz,.txt",req:true,ico:"pairs"},
    {k:"draft",t:"Draft scaffold จาก YaHS / 3D-DNA",ext:".agp / .assembly · ว่างไว้ = สร้างจากศูนย์",acc:".agp,.assembly",req:false,ico:".agp"},
    {k:"truth",t:"คำตอบที่ถูก (ถ้ามี) เพื่อวัดผล",ext:".review.assembly / truth.tsv / .agp",acc:".assembly,.agp,.tsv",req:false,ico:"truth"}],
  refine:[
    {k:"hic",t:"Contact map",ext:".hic / .mcool",acc:".hic,.mcool,.cool",req:true,ico:".hic"},
    {k:"draft",t:"Draft scaffold ที่ใช้สร้าง .hic นั้น",ext:".assembly / .agp",acc:".assembly,.agp",req:true,ico:".asm"},
    {k:"contigs",t:"Contigs (ถ้ามี) เพื่อเขียน .fa",ext:".fa / .fasta / .fai",acc:".fa,.fasta,.fna,.gz,.fai",req:false,ico:".fa"},
    {k:"truth",t:"คำตอบที่ถูก (ถ้ามี) เพื่อวัดผล",ext:".review.assembly / truth.tsv",acc:".assembly,.agp,.tsv",req:false,ico:"truth"}]};
const S={mode:"correct",files:{},job:null,res:null,view:"after",zoom:0,sel:null,filter:"all",env:{},tracks:{},maps:{}};
const TYPES={break:{th:"ตัดรอยต่อ",c:"--t-inter"},join:{th:"ต่อใหม่",c:"--t-move"},flip:{th:"กลับทิศ",c:"--t-flip"},keep:{th:"รอยต่อเดิม",c:"--t-place"}};

/* ---------- upload ---------- */
function renderDrops(){
  $("#drops").innerHTML=SLOTS[S.mode].map(s=>{const f=S.files[s.k];
    return`<div class="drop" data-k="${s.k}">
      <span class="ico">${s.ico}</span>
      <span style="min-width:0"><span class="t">${s.t} ${s.req?'<span class="chip">จำเป็น</span>':'<span class="chip" style="background:var(--surface);color:var(--muted)">ไม่บังคับ</span>'}</span><br>
        <span class="f">${f?(f.err?`<span class="err">${esc(f.err)}</span>`:f.busy?`กำลังอัปโหลด ${esc(f.name)} … ${f.pct||0}%`:`<b>${esc(f.name)}</b>${f.size?" · "+human(f.size):""}${f.demo?" · ข้อมูลตัวอย่าง":""}`):esc(s.ext)}</span>
        <span class="pathrow" hidden style="display:flex;gap:6px;margin-top:6px"><input type="text" class="mono" placeholder="วาง path ของไฟล์ เช่น D:\\hic\\data\\file${esc(s.ico)}" style="flex:1;min-width:0;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);font-size:.78rem"><button class="btn" type="button" data-ok>ใช้</button></span>
      </span>
      <span style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
        <label class="btn" style="cursor:pointer">อัปโหลด<input type="file" accept="${s.acc}" hidden></label>
        <button class="btn" type="button" data-path>ใช้ path</button>
        ${f&&!f.busy?'<button class="btn" type="button" data-clear aria-label="ล้าง">✕</button>':""}
      </span></div>`}).join("");
  $$("#drops .drop").forEach(z=>{const k=z.dataset.k;
    z.querySelector("input[type=file]").onchange=e=>upload(k,e.target.files[0]);
    z.addEventListener("dragover",e=>{e.preventDefault();z.classList.add("over")});
    z.addEventListener("dragleave",()=>z.classList.remove("over"));
    z.addEventListener("drop",e=>{e.preventDefault();z.classList.remove("over");if(e.dataTransfer.files[0])upload(k,e.dataTransfer.files[0])});
    const pr=z.querySelector(".pathrow");
    z.querySelector("[data-path]").onclick=()=>{pr.hidden=!pr.hidden;if(!pr.hidden)pr.querySelector("input").focus()};
    const use=async()=>{const p=pr.querySelector("input").value.trim().replace(/^"|"$/g,"");if(!p)return;
      try{const r=await post("/api/check",{path:p,kind:k});S.files[k]={path:r.path,name:r.name,size:r.size}}catch(e){S.files[k]={err:e.message}}renderDrops()};
    pr.querySelector("[data-ok]").onclick=use;pr.querySelector("input").onkeydown=e=>{if(e.key==="Enter")use()};
    const c=z.querySelector("[data-clear]");if(c)c.onclick=()=>{delete S.files[k];renderDrops()};
  });
}
function upload(k,file){if(!file)return;S.files[k]={name:file.name,size:file.size,busy:true,pct:0};renderDrops();
  const x=new XMLHttpRequest();x.open("POST",`/api/upload?kind=${k}&name=${encodeURIComponent(file.name)}`);
  x.upload.onprogress=e=>{if(e.lengthComputable){S.files[k].pct=Math.round(e.loaded/e.total*100);const el=$(`.drop[data-k="${k}"] .f`);if(el)el.textContent=`กำลังอัปโหลด ${file.name} … ${S.files[k].pct}%`}};
  x.onload=()=>{let j={};try{j=JSON.parse(x.responseText)}catch(e){}
    S.files[k]=x.status===200?{path:j.path,name:j.name,size:j.size}:{err:j.error||"อัปโหลดไม่สำเร็จ"};renderDrops()};
  x.onerror=()=>{S.files[k]={err:"อัปโหลดไม่สำเร็จ"};renderDrops()};x.send(file)}
function setMode(m){S.mode=m;$$('#v-upload .seg button').forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.mode===m)));
  $("#res-field").hidden=$("#scale-field").hidden=m!=="refine";
  for(const k of Object.keys(S.files))if(!SLOTS[m].some(s=>s.k===k))delete S.files[k];renderDrops()}
async function loadEnv(){try{S.env=await api("/api/env")}catch(e){$("#envbox").textContent=e.message;return}
  const e=S.env;if(e.model)$("#model").value=e.model;
  const bits=[];bits.push(e.torch?"✓ PyTorch":"✗ ไม่พบ PyTorch (pip install torch)");
  bits.push(e.model?"✓ โมเดล":"✗ ไม่พบ models/hicscaf.pt");
  bits.push(e.hictk?"✓ อ่าน .hic / .mcool ได้ (hictk)":"✓ อ่าน .hic ได้ (ตัวอ่านสำรองในตัว) · แนะนำ pip install hictkpy");
  if(!e.hictk)bits.push(e.cooler?"✓ อ่าน .mcool ได้":"• อ่าน .mcool ต้องติดตั้ง hictkpy หรือ cooler");
  $("#envbox").innerHTML=bits.map(esc).join("<br>")}
async function loadHistory(){try{const js=await api("/api/jobs");$("#history-card").hidden=!js.length;
  $("#history").innerHTML=js.map(j=>`<div class="file"><span class="ext">${j.state==="done"?"เสร็จ":j.state==="error"?"ผิดพลาด":"กำลังรัน"}</span><span style="min-width:0"><span class="nm">${esc(j.name)}</span><br><span class="ds">${new Date(j.created*1000).toLocaleString("th-TH")} · ${j.params.mode==="refine"?".hic + draft":"contigs + pairs"}</span></span><span class="acts"><button class="btn" type="button" data-j="${j.id}">${j.state==="done"?"ดูผล":"ดูสถานะ"}</button></span></div>`).join("");
  $$("#history [data-j]").forEach(b=>b.onclick=()=>{const j=js.find(x=>x.id===b.dataset.j);openJob(j.id,j.state==="done"?"results":"run")})}catch(e){}}
async function start(){$("#formerr").textContent="";
  const miss=SLOTS[S.mode].filter(s=>s.req&&!(S.files[s.k]&&S.files[s.k].path));
  if(miss.length){$("#formerr").textContent="ยังขาดไฟล์: "+miss.map(s=>s.t).join(", ");return}
  if(Object.values(S.files).some(f=>f.busy)){$("#formerr").textContent="รอให้อัปโหลดเสร็จก่อน";return}
  const p={mode:S.mode,name:$("#jobname").value.trim(),min_prob:+$("#mp").value,resolution:+$("#res").value,scale:+$("#scale").value,
    n_chrom:+$("#nchrom").value||0,model:$("#model").value.trim(),bin_size:+$("#binsize").value||null,juicer_tools:$("#jt").value.trim()||null};
  for(const[k,f]of Object.entries(S.files))if(f.path)p[k]=f.path;
  try{const j=await post("/api/jobs",p);openJob(j.id,"run")}catch(e){$("#formerr").textContent=e.message}}

/* ---------- run ---------- */
let poll=null;
function openJob(id,view){S.job=id;S.res=null;S.maps={};S.tracks={};S.sel=null;S.zoom=0;S.filter="all";location.hash=view;if(view==="run")watch();else route()}
function watch(){clearTimeout(poll);if(!S.job){$("#run-empty").hidden=false;return}
  const tick=async()=>{let j;try{j=await api("/api/jobs/"+S.job)}catch(e){$("#run-stage").textContent=e.message;return}
    $("#run-empty").hidden=true;$("#job-id").textContent="งาน "+j.id;$("#run-title").textContent=j.state==="done"?"ประมวลผลเสร็จแล้ว":j.state==="error"?"เกิดข้อผิดพลาด":"กำลังประมวลผล: "+esc(j.name);
    const st=j.state==="done"?j.steps.length:j.step;const D=j.detail||[];
    $("#stl").innerHTML=j.steps.map((s,i)=>{const d=D[i]||{};const run=i===st&&j.state==="running",err=i===st&&j.state==="error";
      let right="",bar="";
      if(i<st||d.done)right="ใช้เวลา "+dur(d.elapsed);
      else if(run){right=d.frac!=null&&d.frac>=0.995?"กำลังจบขั้นนี้":d.eta!=null?"เหลืออีก ~"+dur(d.eta):"กำลังทำ · "+dur(d.elapsed);
        bar=d.frac!=null?`<span class="mb"><i style="width:${(d.frac*100).toFixed(1)}%"></i></span><span class="mbt">${Math.floor(d.frac*100)}%${d.eta==null&&d.elapsed<3?" · กำลังประมาณเวลา":""}</span>`:`<span class="mb ind"><i></i></span><span class="mbt">ขั้นนี้ประมาณเวลาไม่ได้ ผ่านไปแล้ว ${dur(d.elapsed)}</span>`}
      else if(err)right="ผิดพลาด";
      return`<li class="${i<st?"done":run?"active":""}"><span class="dot">${i<st?"✓":i+1}</span><span style="min-width:0"><span class="nm">${s}</span>${bar?`<span class="mbw">${bar}</span>`:""}</span><span class="tm">${right}</span></li>`}).join("");
    const cur=D[st]||{};const overall=j.state==="done"?1:(Math.max(0,st)+(cur.frac||0))/j.steps.length;
    $("#run-bar").style.width=(overall*100).toFixed(1)+"%";
    $("#run-stage").textContent=j.state==="error"?"ผิดพลาด: "+j.error:j.state==="done"?"เสร็จสิ้น":`ขั้นที่ ${st+1}/${j.steps.length} · ${j.steps[Math.max(0,st)]}`+(cur.eta!=null?` · ขั้นนี้เหลืออีกประมาณ ${dur(cur.eta)}`:"");
    $("#run-time").textContent="ผ่านไปแล้ว "+dur((j.now||Date.now()/1000)-j.created);
    const lg=$("#log");lg.innerHTML=j.log.map(l=>{const m=l.match(/^(\[[^\]]+\])(.*)$/);return m?`<span class="ts">${esc(m[1])}</span>${esc(m[2])}`:esc(l)}).join("\n");lg.scrollTop=1e9;
    $("#see").disabled=j.state!=="done";
    if(j.state==="done"){if(location.hash==="#run")setTimeout(()=>{if(location.hash==="#run")location.hash="results"},900);return}
    if(j.state==="error")return;
    if(location.hash==="#run")poll=setTimeout(tick,1000)};
  tick()}

/* ---------- results ---------- */
let LUT=null;
function hex(h){h=h.replace("#","");const n=parseInt(h,16);return[(n>>16)&255,(n>>8)&255,n&255]}
function buildLUT(){const lo=hex(css("--map-lo")),hi=hex(css("--map-hi"));LUT=new Uint8ClampedArray(768);for(let i=0;i<256;i++){const t=Math.pow(i/255,1.1);for(let k=0;k<3;k++)LUT[i*3+k]=lo[k]+(hi[k]-lo[k])*t}}
async function getMap(which,scaf,n){const key=which+scaf+"_"+n;if(!S.maps[key]){const m=await api(`/api/jobs/${S.job}/map?which=${which}&scaf=${scaf}&n=${n}`);
  const raw=atob(m.data);const a=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)a[i]=raw.charCodeAt(i);m.px=a;S.maps[key]=m}return S.maps[key]}
function paint(cv,m){const n=m.n;cv.width=n;cv.height=n;const ctx=cv.getContext("2d");const img=ctx.createImageData(n,n),d=img.data;
  for(let i=0;i<n*n;i++){const v=m.px[i]*3;d[i*4]=LUT[v];d[i*4+1]=LUT[v+1];d[i*4+2]=LUT[v+2];d[i*4+3]=255}ctx.putImageData(img,0,0);return ctx}
async function getTracks(which){if(!S.tracks[which])S.tracks[which]=await api(`/api/jobs/${S.job}/tracks?which=${which}`);return S.tracks[which]}
const dcol=(d,alt)=>`hsl(${((d*137.508+18)%360).toFixed(0)} 52% ${alt?50:63}%)`;

async function renderResults(){
  if(!S.job){try{const js=await api("/api/jobs");const d=js.find(j=>j.state==="done");if(d)S.job=d.id}catch(e){}}
  $("#res-empty").hidden=!!S.job;$("#res-body").hidden=!S.job;if(!S.job)return;
  try{if(!S.res)S.res=await api(`/api/jobs/${S.job}/result`)}catch(e){$("#res-empty").hidden=false;$("#res-empty").textContent=e.message;$("#res-body").hidden=true;return}
  const st=await api("/api/jobs/"+S.job);S.status=st;
  $("#res-id").textContent="งาน "+st.id+" · เสร็จแล้ว";$("#res-title").textContent=st.params.name||"ผลการแก้ scaffold";
  const P=st.params,fn=p=>p?esc(p.split(/[\\/]/).pop()):"";
  $("#res-meta").innerHTML=[P.mode==="refine"?`<span>contact map <b>${fn(P.hic)}</b></span>`:`<span>pairs <b>${fn(P.pairs)}</b></span>`,P.draft?`<span>draft <b>${fn(P.draft)}</b></span>`:`<span><b>สร้างจากศูนย์</b> (ไม่มี draft)</span>`,`<span>ชิ้น contig <b>${num(S.res.components)}</b></span>`,`<span>bin <b>${num(S.res.bin_size)} bp</b></span>`].join("");
  $("#thr").value=S.res.min_prob;$("#thr-o").textContent=(+S.res.min_prob).toFixed(2);$("#nc").value=S.res.n_chrom||0;
  renderTiles();renderEdits();renderEval();renderFiles();
  await renderMaps();
}
function renderTiles(){const r=S.res,b=r.before,a=r.after;
  $("#tiles").innerHTML=`
  <div class="tile"><span class="k">จำนวน scaffold</span><span class="v">${num(a.scaffolds)}</span><span class="s">ก่อนแก้ ${num(b.scaffolds)}</span></div>
  <div class="tile"><span class="k">Scaffold N50</span><span class="v">${bp(a.N50)}</span><span class="s">ก่อนแก้ ${bp(b.N50)} · ใหญ่สุด ${bp(a.largest)}</span></div>
  <div class="tile ok"><span class="k">AI แก้ให้</span><span class="v">${num(r.counts.break+r.counts.join+r.counts.flip)} <small>จุด</small></span><span class="s">ตัด ${r.counts.break} · ต่อใหม่ ${r.counts.join} · กลับทิศ ${r.counts.flip}</span></div>
  <div class="tile warn"><span class="k">ควรตรวจเองใน Juicebox</span><span class="v">${num(r.review)} <small>จุด</small></span><span class="s">ความมั่นใจใกล้เกณฑ์ ${(+r.min_prob).toFixed(2)}</span></div>`+
  (r.chroms?`<div class="tile ok"><span class="k">จัดกลุ่มโครโมโซม</span><span class="v">${num(r.chroms.list.length)} <small>กลุ่ม</small></span><span class="s">รวม ${bp(r.chroms.placed)} (${(100*r.chroms.placed_frac).toFixed(1)}% ของจีโนม) · ไม่ได้จัดกลุ่ม ${num(r.chroms.unplaced)} ชิ้น</span><span class="s">${r.chroms.list.map(c=>esc(c.name)+" "+bp(c.length)).join(" · ")}</span></div>`:"")}
async function renderMaps(){
  $$('#v-results .seg button').forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view===S.view)));
  const m=await getMap(S.view,-1,600);const cv=$("#gmap");const ctx=paint(cv,m);const sc=m.n/m.total;
  ctx.strokeStyle=css("--map-grid");ctx.lineWidth=.6;ctx.beginPath();
  m.bounds.slice(1).forEach(b=>{const p=b*sc;if(p<1)return;ctx.moveTo(p,0);ctx.lineTo(p,m.n);ctx.moveTo(0,p);ctx.lineTo(m.n,p)});ctx.stroke();
  if(S.zoom<m.bounds.length){const a=m.bounds[S.zoom]*sc,e=(S.zoom+1<m.bounds.length?m.bounds[S.zoom+1]:m.total)*sc;ctx.strokeStyle=css("--map-sel");ctx.lineWidth=2;ctx.strokeRect(a,a,Math.max(2,e-a),Math.max(2,e-a))}
  S.gm=m;
  const t=await getTracks(S.view);S.tr=t;renderKaryo();renderZoomSel();await renderZoom();
}
function renderKaryo(){const t=S.tr;const maxL=Math.max(...t.rows.map(r=>r.len));
  $("#karyo").innerHTML=t.rows.map((r,s)=>{const nf=r.parts.filter(p=>p.flip||p.foreign).length;
    return`<div class="krow${s===S.zoom?" sel":""}" data-s="${s}" role="button" tabindex="0" title="${esc(r.name)} · ${bp(r.len)}"><span class="nm">${s+1}</span><span style="min-width:0"><span class="ktrack" style="width:${(r.len/maxL*100).toFixed(2)}%">${r.parts.map((p,k)=>`<span class="seg-c${p.flip?" flip":""}${p.foreign?" foreign":""}" style="flex:${p.len} 0 0;background-color:${dcol(p.d,k%2)}"></span>`).join("")}</span></span><span class="cnt${nf?" has":""}">${nf||""}</span></div>`}).join("")+
    (t.more?`<p class="note" style="margin:6px 0 0 52px">และอีก ${num(t.more)} scaffold ขนาดเล็ก</p>`:"");
  $("#k-sub").textContent=`${num(t.count)} scaffold · สีของแต่ละชิ้น = scaffold ใน draft ที่ชิ้นนั้นมาจาก`}
function renderZoomSel(){$("#z-sel").innerHTML=S.tr.rows.map((r,s)=>`<option value="${s}">${s+1}. ${esc(r.name)} · ${bp(r.len)}</option>`).join("");if(S.zoom>=S.tr.rows.length)S.zoom=0;$("#z-sel").value=S.zoom}
async function renderZoom(){const row=S.tr.rows[S.zoom];if(!row)return;
  const m=await getMap(S.view,S.zoom,480);const ctx=paint($("#zmap"),m);const sc=m.n/m.total;S.zm=m;S.zrow=row;
  // contig boundaries
  ctx.strokeStyle=css("--map-grid");ctx.lineWidth=.5;ctx.setLineDash([2,3]);ctx.beginPath();let pos=0;row.parts.forEach((p,k)=>{if(k){const q=pos*sc;ctx.moveTo(q,q-12);ctx.lineTo(q,q+12);ctx.moveTo(q-12,q);ctx.lineTo(q+12,q)}pos+=p.len});ctx.stroke();ctx.setLineDash([]);
  // highlight parts touched by edits
  const touched=new Map();S.res.edits.forEach(e=>{[e.a,e.b].forEach(i=>{if(!touched.has(i)||e.id===S.sel)touched.set(i,e)})});
  pos=0;row.parts.forEach(p=>{const e=touched.get(p.i);if(e&&(e.status==="applied"||e.id===S.sel)){const a=pos*sc,w=p.len*sc;ctx.strokeStyle=css(e.status==="review"?"--warn":TYPES[e.type].c);ctx.lineWidth=e.id===S.sel?3:1.4;if(S.view==="after")ctx.setLineDash([4,3]);ctx.strokeRect(a,a,Math.max(w,2),Math.max(w,2));ctx.setLineDash([])}pos+=p.len});
  const selE=S.sel!=null?S.res.edits.find(e=>e.id===S.sel):null;
  $("#ztrack").innerHTML=row.parts.map((p,k)=>{const lbl=p.len/row.len>0.08?(p.st<0?"◀ ":"")+esc(p.name)+(p.st>0?" ▶":""):"";const hl=selE&&(selE.a===p.i||selE.b===p.i);
    return`<span class="seg-c${p.flip?" flip":""}${p.foreign?" foreign":""}${hl?" hl":""}" data-i="${p.i}" style="flex:${p.len} 0 0;background-color:${dcol(p.d,k%2)}">${lbl}</span>`}).join("");
  $("#z-len").textContent=bp(row.len);
  $("#z-title").textContent=`ซูม ${row.name} · ${S.view==="before"?"ก่อนแก้ (draft)":"หลัง AI แก้"}`;
  const nf=row.parts.filter(p=>p.flip).length,nm=row.parts.filter(p=>p.foreign).length;
  $("#z-sub").textContent=`${num(row.parts.length)} ชิ้น · ${bp(row.len)}`+(S.view==="after"?` · กลับทิศเทียบ draft ${nf} · ย้ายมาจาก scaffold อื่น ${nm}`:"");
  renderDetail()}
function renderDetail(){const host=$("#detail");const e=S.sel!=null?S.res.edits.find(x=>x.id===S.sel):null;
  if(!e){const ids=new Set(S.zrow.parts.map(p=>p.i));const here=S.res.edits.filter(x=>ids.has(x.a)||ids.has(x.b));
    host.innerHTML=`<div class="empty">เลือกรายการจากตารางด้านล่าง หรือคลิกชิ้น contig บนแถบสี เพื่อดูรายละเอียด<br><br>${here.length?`scaffold นี้มี ${here.length} รายการที่เกี่ยวข้อง <button class="btn" id="d-first" type="button" style="margin-top:8px">ดูรายการแรก</button>`:"scaffold นี้ไม่มีรายการแก้"}</div>`;
    const b=$("#d-first");if(b)b.onclick=()=>selectEdit(here[0].id);return}
  const T=TYPES[e.type];const[sc,sl]=stat(e);
  host.innerHTML=`<div style="display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap"><span class="type"><i style="background:var(${T.c})"></i>${T.th}</span><span class="st ${sc}">${sl}</span></div>
  <h3 class="mono" style="font-size:1rem;overflow-wrap:anywhere">${esc(e.contigs[0])} ↔ ${esc(e.contigs[1])}</h3><p>${esc(e.detail)}</p>
  <dl class="dl"><dt>EdgeGNN</dt><dd>P = ${e.prob.toFixed(3)} · เกณฑ์ ${(+S.res.min_prob).toFixed(2)}</dd><dt>JunctionCNN</dt><dd>${e.cnn==null?"ไม่เกี่ยว (ไม่ใช่รอยต่อใน draft)":"P(รอยต่อเดิมถูก) = "+e.cnn.toFixed(3)}</dd></dl>
  <p class="note">ดูจุดนี้ใน Juicebox: เปิด .hic ของ draft แล้ว Import Map Assembly เป็น corrected.assembly แล้วค้นชื่อ contig ข้างบน</p>`}
function stat(e){return e.status==="applied"?["ok","แก้แล้ว"]:["warn","ควรตรวจ"]}
function selectEdit(id){S.sel=id;const e=S.res.edits.find(x=>x.id===id);
  const r=S.tr.rows.findIndex(r=>r.parts.some(p=>p.i===e.a||p.i===e.b));if(r>=0)S.zoom=r;
  $("#z-sel").value=S.zoom;renderKaryo();renderMaps();$$("#edits tr").forEach(t=>t.classList.toggle("sel",+t.dataset.id===id));
  $("#zoomcard").scrollIntoView({behavior:"smooth",block:"start"})}
function renderEdits(){const E=S.res.edits;
  const fl=[["all","ทั้งหมด",()=>true],["break","ตัดรอยต่อ",e=>e.type==="break"&&e.status==="applied"],["join","ต่อใหม่",e=>e.type==="join"&&e.status==="applied"],["flip","กลับทิศ",e=>e.type==="flip"],["review","ควรตรวจ",e=>e.status==="review"]];
  $("#tbar").innerHTML=fl.map(([k,l,f])=>`<button type="button" data-f="${k}" aria-pressed="${S.filter===k}">${l}<span class="c">${E.filter(f).length}</span></button>`).join("");
  $$("#tbar button").forEach(b=>b.onclick=()=>{S.filter=b.dataset.f;renderEdits()});
  const rows=E.filter(fl.find(x=>x[0]===S.filter)[2]);const LIM=400;const th=+S.res.min_prob;
  $("#edits").innerHTML=rows.slice(0,LIM).map(e=>{const[sc,sl]=stat(e);const T=TYPES[e.type];
    return`<tr data-id="${e.id}" class="${S.sel===e.id?"sel":""}" tabindex="0"><td><span class="type"><i style="background:var(${T.c})"></i>${T.th}</span></td><td class="mono" style="font-size:.78rem;overflow-wrap:anywhere;max-width:260px">${esc(e.contigs[0])}<br>${esc(e.contigs[1])}</td><td style="min-width:200px">${esc(e.detail)}</td><td><span class="conf"><span class="cb"><i style="width:${(e.prob*100).toFixed(0)}%"></i><b style="left:${(th*100).toFixed(0)}%"></b></span><span>${e.prob.toFixed(2)}</span></span></td><td class="mono" style="font-size:.8rem">${e.cnn==null?"–":e.cnn.toFixed(2)}</td><td><span class="st ${sc}">${sl}</span></td></tr>`}).join("")||`<tr><td colspan="6" class="muted">ไม่มีรายการ</td></tr>`;
  $("#edits-more").textContent=rows.length>LIM?`แสดง ${LIM} จาก ${num(rows.length)} รายการ ทั้งหมดอยู่ในไฟล์ edits.tsv`:"";
  $$("#edits tr[data-id]").forEach(r=>{r.onclick=()=>selectEdit(+r.dataset.id);r.onkeydown=e=>{if(e.key==="Enter")selectEdit(+r.dataset.id)}})}
function renderEval(){const ev=S.res.eval;$("#eval-card").hidden=!ev;if(!ev)return;
  $("#eval-sub").textContent="วัดจากไฟล์คำตอบที่ใส่มา ตัวเลขสีเขียวคือดีกว่า";
  const R=[["ความแม่นของรอยต่อ (precision)","precision",x=>(x*100).toFixed(1)+"%","hi"],["รอยต่อที่หาเจอ (recall)","recall",x=>(x*100).toFixed(1)+"%","hi"],["F1","F1",x=>x.toFixed(3),"hi"],["misjoin ข้ามโครโมโซม","misjoin_interchrom",num,"lo"],["misjoin ย้ายตำแหน่ง","misjoin_relocation",num,"lo"],["ลำดับผิดเฉพาะที่","local_misorder",num,"lo"],["ทิศผิด","orientation_error",num,"lo"],["จำนวน scaffold","scaffolds",num,"none"],["N50","N50",bp,"hi"]];
  let h=`<thead><tr><th>ตัวชี้วัด</th><th>ก่อน (draft)</th><th class="ours">หลัง (AI)</th></tr></thead><tbody>`;
  R.forEach(([n,k,f,dir])=>{const a=ev.before[k],b=ev.after[k];const best=dir==="hi"?Math.max(a,b):dir==="lo"?Math.min(a,b):null;
    h+=`<tr><td>${n}</td><td class="${best!=null&&a===best&&a!==b?"best":""}">${f(a)}</td><td class="ours ${best!=null&&b===best&&a!==b?"best":""}">${f(b)}</td></tr>`});
  $("#cmp").innerHTML=h+"</tbody>"}
function renderFiles(){const D={"corrected.assembly":"สำหรับ Juicebox Assembly Tools ใช้คู่กับ .hic ของ draft","corrected.agp":"AGP 2.1 สำหรับส่ง NCBI หรือใช้กับเครื่องมืออื่น","corrected.fa":"ลำดับเบสของ scaffold ใหม่","edits.tsv":"รายการแก้ทุกจุดพร้อมคะแนน (เปิดใน Excel ได้)"};
  $("#files").innerHTML=S.res.files.map(f=>`<div class="file"><span class="ext">${esc(f.name.split(".").pop())}</span><span style="min-width:0"><span class="nm">${esc(f.name)}</span> <span class="note">${human(f.size)}</span><br><span class="ds">${D[f.name]||""}</span></span><span class="acts"><a class="btn primary" href="/api/jobs/${S.job}/file/${encodeURIComponent(f.name)}" download>ดาวน์โหลด</a></span></div>`).join("")+
   (S.res.files.some(f=>f.name==="corrected.fa")?"":`<p class="note">ไม่มี corrected.fa เพราะไม่ได้ใส่ contigs แบบ FASTA</p>`);
  $("#outdir").textContent="ไฟล์ทั้งหมดถูกบันทึกไว้ในโฟลเดอร์งานบนเครื่องนี้ด้วย ("+(S.status?S.status.id:"")+")"}

/* ---------- interactions ---------- */
function bindResults(){
  $$('#v-results .seg button').forEach(b=>b.onclick=()=>{S.view=b.dataset.view;renderMaps()});
  document.addEventListener("keydown",e=>{if(location.hash!=="#results"||e.target.matches("input,select,textarea"))return;if(e.key==="b"||e.key==="B"){S.view=S.view==="before"?"after":"before";renderMaps()}});
  $("#thr").oninput=e=>{$("#thr-o").textContent=(+e.target.value).toFixed(2)};
  $("#thr").onchange=e=>regroup({min_prob:+e.target.value,n_chrom:+$("#nc").value||0},"กำลังประกอบใหม่…");
  async function regroup(body,note){$("#thr-note").textContent=note;try{S.res=await post(`/api/jobs/${S.job}/threshold`,body);S.maps={};delete S.tracks.after;S.sel=null;renderTiles();renderEdits();renderEval();renderFiles();await renderMaps();$("#thr-note").textContent="ประกอบใหม่และเขียนไฟล์ใหม่แล้ว"}catch(err){$("#thr-note").textContent=err.message}}
  $("#nc-go").onclick=()=>regroup({min_prob:+$("#thr").value,n_chrom:+$("#nc").value||0},"กำลังจัดกลุ่มโครโมโซมใหม่…");
  const gm=$("#gmap");const at=(cv,m,e)=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)/r.width*m.total,(e.clientY-r.top)/r.height*m.total]};
  const sOf=(m,x)=>{let s=0;for(let k=0;k<m.bounds.length;k++)if(m.bounds[k]<=x)s=k;return s};
  gm.onmousemove=e=>{if(!S.gm)return;const[x,y]=at(gm,S.gm,e);const a=sOf(S.gm,x),b=sOf(S.gm,y);$("#g-read").innerHTML=`X: <b>${esc(S.gm.names[a])}</b> ${bp(x-S.gm.bounds[a])} · Y: <b>${esc(S.gm.names[b])}</b> ${bp(y-S.gm.bounds[b])}`};
  gm.onclick=e=>{if(!S.gm)return;const s=sOf(S.gm,at(gm,S.gm,e)[0]);if(s>=S.tr.rows.length){toast("scaffold นี้เล็กเกินกว่าจะแสดงในรายการ");return}S.zoom=s;S.sel=null;renderKaryo();renderMaps()};
  const partAt=x=>{let pos=0;for(const p of S.zrow.parts){if(x<pos+p.len)return[p,x-pos];pos+=p.len}return[S.zrow.parts[S.zrow.parts.length-1],0]};
  const zm=$("#zmap");zm.onmousemove=e=>{if(!S.zm)return;const[x,y]=at(zm,S.zm,e);const[p]=partAt(x),[q]=partAt(y);$("#z-read").innerHTML=`X: <b>${bp(x)}</b> ${esc(p.name)} · Y: <b>${bp(y)}</b> ${esc(q.name)}`};
  zm.onclick=e=>{if(!S.zm)return;clickPart(partAt(at(zm,S.zm,e)[0])[0].i)};
  $("#ztrack").onclick=e=>{const s=e.target.closest("[data-i]");if(s)clickPart(+s.dataset.i)};
  $("#ztrack").onmousemove=e=>{const s=e.target.closest("[data-i]");if(!s)return;const p=S.zrow.parts.find(p=>p.i===+s.dataset.i);$("#z-read").innerHTML=`<b>${esc(p.name)}</b> · ${bp(p.len)} · ${p.st>0?"+":"−"}${p.flip?" · กลับทิศเทียบ draft":""}${p.foreign?" · ย้ายมาจาก scaffold อื่น":""}`};
  $("#karyo").onclick=e=>{const r=e.target.closest(".krow[data-s]");if(!r)return;S.zoom=+r.dataset.s;S.sel=null;renderKaryo();renderMaps()};
  $("#karyo").onkeydown=e=>{if(e.key==="Enter"){const r=e.target.closest(".krow[data-s]");if(r)r.click()}};
  $("#z-sel").onchange=e=>{S.zoom=+e.target.value;S.sel=null;renderKaryo();renderMaps()};
  $("#z-prev").onclick=()=>{S.zoom=(S.zoom-1+S.tr.rows.length)%S.tr.rows.length;S.sel=null;renderKaryo();renderMaps()};
  $("#z-next").onclick=()=>{S.zoom=(S.zoom+1)%S.tr.rows.length;S.sel=null;renderKaryo();renderMaps()};
}
function clickPart(i){const e=S.res.edits.find(x=>(x.a===i||x.b===i)&&x.status==="applied")||S.res.edits.find(x=>x.a===i||x.b===i);if(e)selectEdit(e.id);else{S.sel=null;renderZoom();toast("ชิ้นนี้ไม่มีการแก้")}}
let tt;function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("show");clearTimeout(tt);tt=setTimeout(()=>t.classList.remove("show"),3000)}

function route(){let v=(location.hash||"#upload").slice(1);if(!["upload","run","results"].includes(v))v="upload";
  $$(".view[id^=v-]").forEach(s=>s.hidden=s.id!=="v-"+v);
  $$(".steps a").forEach(a=>{if(a.dataset.v===v)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current")});
  if(v==="upload")loadHistory();if(v==="run")watch();if(v==="results")renderResults().catch(e=>toast(e.message));window.scrollTo(0,0)}

$$('#v-upload .seg button').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
$("#mp").oninput=e=>{$("#mp-o").textContent=(+e.target.value).toFixed(2)};
$("#start").onclick=start;
$("#demo").onclick=async()=>{try{const d=await post("/api/demo");setMode("correct");for(const k of["contigs","pairs","draft","truth"])S.files[k]={path:d[k],name:d[k].split(/[\\/]/).pop(),size:0,demo:true};
  $("#jobname").value=$("#jobname").value||"ทดลองกับจีโนมจำลอง";renderDrops();toast("ใส่ข้อมูลตัวอย่างแล้ว กดเริ่มประมวลผลได้เลย")}catch(e){toast(e.message)}};
$("#see").onclick=()=>{location.hash="results"};
$("#back-upload").onclick=()=>{location.hash="upload"};
const retheme=()=>{buildLUT();S.maps={};if(location.hash==="#results"&&S.res)renderMaps()};
try{matchMedia("(prefers-color-scheme: dark)").addEventListener("change",retheme)}catch(e){}
buildLUT();bindResults();setMode("correct");loadEnv();
window.addEventListener("hashchange",route);route();
})();
