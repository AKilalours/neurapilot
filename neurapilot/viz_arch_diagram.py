"""
NeuraPilot — Paper Architecture Diagram (AI-style overview)
Generates full research paper workflow diagrams from AI response text.
"""
from __future__ import annotations
import re, textwrap
from typing import Any

BG="#0d1117"; PANEL="#0f1520"; PANEL2="#111827"
BLUE="#3b82f6"; AMBER="#f59e0b"; GREEN="#10b981"
PURPLE="#8b5cf6"; PINK="#ec4899"; CYAN="#06b6d4"
ORANGE="#f97316"; TEAL="#14b8a6"; ROSE="#f43f5e"
WHITE="#f1f5f9"; MUTED="#94a3b8"; DIM="#475569"
PALETTE=[BLUE,AMBER,GREEN,PURPLE,PINK,CYAN,ORANGE,TEAL,ROSE]

def _c(t):
    if not t: return ""
    t=re.sub(r'\*{1,3}([^*]+)\*{1,3}',r'\1',str(t))
    t=re.sub(r'\[S\d+\]|\[chunk:\d+\]|\(page=\d+\)','',t)
    return re.sub(r'\s+',' ',t).strip(' :•–—\t→')

def _w(txt,width=30,lines=3):
    return '\n'.join(textwrap.wrap(str(txt),width)[:lines])

def parse_structure(content: str) -> dict:
    """Smart parser: extracts title, problem, pipeline stages, contributions, output."""
    lines=[l.strip() for l in content.splitlines() if l.strip()]

    # ── Title: first heading, or caller-supplied topic ─────────────────────
    title=""
    for line in lines[:10]:
        # heading like "# Title" or "## 3) Model pipeline..."
        m=re.match(r'^#{1,4}\s*(?:\d+[.)]\s*)?(.+)',line)
        if m: title=_c(m.group(1))[:65]; break
        # standalone bold
        m=re.match(r'^\*\*([^*]{6,65})\*\*$',line)
        if m: title=_c(m.group(1))[:65]; break
    if not title:
        # skip section-number prefix like "3) Model pipeline..."
        first=_c(lines[0]) if lines else ""
        first=re.sub(r'^\d+[.)]\s+','',first)
        title=first[:65]

    # ── Problem statement ──────────────────────────────────────────────────
    problem=""
    kw=["problem:","challenge:","goal:","aim:","proposes","address",
        "however","limitation","difficult","despite","this paper","low-res"]
    for line in lines[:30]:
        lw=line.lower()
        if any(w in lw for w in kw):
            p=_c(line)
            if 20<len(p)<200: problem=p; break

    # ── Numbered pipeline stages ───────────────────────────────────────────
    stages=[]
    i=0
    while i<len(lines):
        line=lines[i]
        # Match "1." "2." "3." etc. but NOT heading-style "3) Section title"
        m=re.match(r'^(\d+)[.]\s+(.+)',line)
        if m:
            raw=_c(m.group(2))
            # Prefer the first bold phrase as name, else first segment before →
            bold=re.search(r'\*\*([^*]{3,40})\*\*',raw)
            if bold:
                name=_c(bold.group(1))
            else:
                # Take text before first →, +, or : as the name
                seg=re.split(r'[→+:]',raw,1)[0]
                name=_c(seg)[:40]
                if len(name)<4: name=raw[:40]

            items=[]
            # Sub-bullets immediately after
            j=i+1
            while j<len(lines) and re.match(r'^[•\-*]\s+',lines[j]):
                items.append(_c(re.sub(r'^[•\-*]\s+','',lines[j]))[:55])
                j+=1
            # No sub-bullets → split the line on → + to get items
            if not items:
                parts=re.split(r'[→+]',raw)
                for p in parts:
                    p=_c(p)
                    if 4<len(p)<55: items.append(p)
            stages.append({"name":name,"items":items[:3]})
            i=j; continue
        i+=1
    stages=stages[:5]

    # ── Bold key contributions ─────────────────────────────────────────────
    contribs=[]
    seen=set()
    for m in re.finditer(r'\*\*([^*]{4,60})\*\*\s*[:\-–]?\s*([^\n*]{0,150})?',content):
        t=_c(m.group(1)); d=_c(m.group(2) or "")[:90]
        k=t.lower()[:22]
        if k not in seen and len(t)>3 and t.lower() not in title.lower():
            contribs.append({"name":t[:48],"desc":d}); seen.add(k)
        if len(contribs)>=5: break

    # ── Section headings as stages if no numbered items found ──────────────
    if len(stages)<2:
        for line in lines:
            m=re.match(r'^#{1,4}\s+(.+)',line)
            if m:
                name=_c(m.group(1))[:45]
                if name.lower() not in title.lower() and len(name)>3:
                    stages.append({"name":name,"items":[]})
            if len(stages)>=5: break

    # ── Output ─────────────────────────────────────────────────────────────
    output=""
    for line in lines:
        lw=line.lower()
        if any(w in lw for w in ["output:","result:","produces","final prediction","predicts","generates"]):
            output=_c(line)[:120]; break

    return {"title":title,"problem":problem,
            "stages":stages,"contribs":contribs,"output":output}

# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _rect(ax,x,y,w,h,color,alpha=0.14,lw=2.0):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x+0.08,y-0.08),w,h,
        boxstyle="round,pad=0.12",lw=0,facecolor="#000018",alpha=0.5,zorder=2))
    ax.add_patch(FancyBboxPatch((x-0.06,y-0.06),w+0.12,h+0.12,
        boxstyle="round,pad=0.12",lw=0,facecolor=color,alpha=alpha*0.7,zorder=2))
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle="round,pad=0.12",lw=lw,edgecolor=color,facecolor=PANEL,zorder=3))

def _hdr(ax,x,y,w,title,color,fs=9.5,icon=""):
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x,y-0.33),w,0.35,
        boxstyle="round,pad=0.04",lw=0,facecolor=color,alpha=0.88,zorder=4))
    lbl=f"{icon}  {title}" if icon else title
    ax.text(x+0.18,y-0.14,lbl,ha='left',va='center',fontsize=fs,
            fontweight='bold',color='white',
            path_effects=[pe.withStroke(linewidth=2,foreground=color)],zorder=5)

def _items(ax,x,y,items,fs=8.0,max_i=4):
    for i,item in enumerate(items[:max_i]):
        ax.text(x+0.22,y-i*0.40,f"▸  {item[:50]}",
                ha='left',va='center',fontsize=fs,color=MUTED,zorder=5)

def _arw_h(ax,x1,y,x2,color=AMBER,lw=2.2):
    ax.annotate("",xy=(x2,y),xytext=(x1,y),
        arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,
            mutation_scale=22,connectionstyle="arc3,rad=0.0"),zorder=9)

def _arw_v(ax,x,y1,y2,color=AMBER,lw=2.2):
    ax.annotate("",xy=(x,y2),xytext=(x,y1),
        arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,
            mutation_scale=20,connectionstyle="arc3,rad=0.0"),zorder=9)

def _lbl(ax,x,y,txt,color=WHITE,fs=9,bold=False,align='center'):
    import matplotlib.patheffects as pe
    ax.text(x,y,txt,ha=align,va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',color=color,
            path_effects=[pe.withStroke(linewidth=3,foreground=BG)],zorder=10)

def _badge(ax,cx,cy,num,color):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((cx,cy),0.27,color=color,zorder=6,clip_on=False))
    ax.text(cx,cy,str(num),ha='center',va='center',
            fontsize=13.5,fontweight='bold',color=BG,zorder=7)


# ─── Main renderer ─────────────────────────────────────────────────────────

def draw_paper_overview(topic: str, content: str) -> Any:
    """
    Generates the AI-style full paper overview diagram:
    
    ┌─ Title ───────────────────────────────────────────────────────┐
    │  Problem / Goal                                               │
    ├───────────────────────────────────────────────────────────────┤
    │  ┌─ Stage 1 ─┐  →  ┌─ Stage 2 ─┐  →  ┌─ Stage 3 ─┐        │
    │  └───────────┘      └───────────┘      └───────────┘        │
    ├───────────────────────────────────────────────────────────────┤
    │  ┌─ Contrib A ─┐   ┌─ Contrib B ─┐   ┌─ Contrib C ─┐       │
    │  └─────────────┘   └─────────────┘   └─────────────┘       │
    ├───────────────────────────────────────────────────────────────┤
    │  ▶ OUTPUT ─────────────────────────────────────────────────   │
    └───────────────────────────────────────────────────────────────┘
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    data     = parse_structure(content)
    title    = data["title"] or topic
    problem  = data["problem"]
    stages   = data["stages"]
    contribs = data["contribs"]
    output   = data["output"]

    # Ensure enough stages
    if len(stages)<2:
        if contribs:
            stages=[{"name":c["name"],"items":[c["desc"]] if c["desc"] else []}
                    for c in contribs[:4]]
        else:
            stages=[{"name":"Input","items":["Multi-view images","Camera parameters"]},
                    {"name":"Feature Extraction","items":["Backbone + FPN","2D feature maps"]},
                    {"name":"3D Lifting","items":["Voxel query","View transformation"]},
                    {"name":"Output","items":["Occupancy prediction","Semantic labels"]}]

    n_s  = min(len(stages),5)
    FW,FH= 20,14.5
    M    = 0.45

    fig,ax=plt.subplots(figsize=(20,14.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0,FW); ax.set_ylim(0,FH)

    # ── OUTER FRAME ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((M,M),FW-2*M,FH-2*M,
        boxstyle="round,pad=0.2",lw=1.2,
        edgecolor="#1e3a5f",facecolor=PANEL+"cc",zorder=1))

    # ── TITLE SECTION ───────────────────────────────────────────────────────
    title_y=FH-M-0.08
    # Taller title band to avoid overlap
    ax.add_patch(FancyBboxPatch((M+0.1,title_y-1.55),FW-2*M-0.2,1.55,
        boxstyle="round,pad=0.1",lw=0,facecolor=BLUE,alpha=0.18,zorder=2))
    ax.text(FW/2,title_y-0.42,title[:60]+"…" if len(title)>60 else title,
            ha='center',va='center',
            fontsize=15,fontweight='bold',color=WHITE,
            path_effects=[pe.withStroke(linewidth=6,foreground=BG)],zorder=5)
    if problem:
        prob_short = problem[:120]+"…" if len(problem)>120 else problem
        ax.text(FW/2,title_y-1.10,_w(prob_short,80,1),
                ha='center',va='center',fontsize=9.5,color=MUTED,zorder=5)
    ax.plot([M+0.2,FW-M-0.2],[title_y-1.58,title_y-1.58],
            color=BLUE,lw=0.8,alpha=0.35,zorder=4)

    # ── PIPELINE SECTION ────────────────────────────────────────────────────
    pipe_top = title_y - 1.78
    pipe_h   = 3.8
    pipe_bot = pipe_top - pipe_h

    # Section label
    ax.text(M+0.35,pipe_top-0.22,"PIPELINE  ›  HOW IT WORKS",
            ha='left',va='center',fontsize=9,fontweight='bold',color=DIM,zorder=5)

    BW  = (FW-2*M-0.6-(n_s-1)*0.5)/n_s
    GAP = 0.5
    bx0 = M+0.3

    stage_cx=[]
    for i,s in enumerate(stages[:n_s]):
        bx  = bx0+i*(BW+GAP)
        by  = pipe_bot+0.1
        col = PALETTE[i%len(PALETTE)]

        _rect(ax,bx,by,BW,pipe_h-0.35,col,alpha=0.15,lw=2.2)

        # Header
        hdr_short=s["name"] if len(s["name"])<=30 else s["name"][:28]+"…"
        _hdr(ax,bx,by+pipe_h-0.37,BW,hdr_short,col,fs=7.8)

        # Step badge (top right)
        _badge(ax,bx+BW-0.35,by+pipe_h-0.62,i+1,col)

        # Items
        items=s["items"]
        if items:
            _items(ax,bx,by+pipe_h-0.90,items,fs=7.2,max_i=4)
        else:
            ax.text(bx+BW/2,by+pipe_h*0.42,_w(s["name"],18,2),
                    ha='center',va='center',fontsize=8.5,color=MUTED,
                    style='italic',linespacing=1.5,zorder=5)

        stage_cx.append(bx+BW/2)

        # Arrow between stages
        if i<n_s-1:
            ax1=bx+BW+0.04; ax2=bx+BW+GAP-0.04
            ay =by+pipe_h*0.5
            _arw_h(ax,ax1,ay,ax2,color=AMBER,lw=2.3)
            ax.text((ax1+ax2)/2,ay+0.16,"→",ha='center',va='bottom',
                    fontsize=10,color=AMBER,fontweight='bold',zorder=9)

    ax.plot([M+0.2,FW-M-0.2],[pipe_bot-0.04,pipe_bot-0.04],
            color=DIM,lw=0.5,alpha=0.3,zorder=3)

    # ── CONTRIBUTIONS SECTION ───────────────────────────────────────────────
    n_c = min(len(contribs),4)
    if n_c>0:
        contrib_top = pipe_bot-0.25
        CH   = 2.8
        contrib_bot = contrib_top - CH

        ax.text(M+0.35,contrib_top-0.22,"KEY CONTRIBUTIONS  ›  INNOVATIONS",
                ha='left',va='center',fontsize=9,fontweight='bold',color=DIM,zorder=5)

        CW  = (FW-2*M-0.6-(n_c-1)*0.5)/n_c
        CGAP= 0.5

        for i,c in enumerate(contribs[:n_c]):
            cx2 = bx0+i*(CW+CGAP)
            col = PALETTE[(n_s+i)%len(PALETTE)]
            _rect(ax,cx2,contrib_bot+0.1,CW,CH-0.2,col,alpha=0.13,lw=1.9)
            hdr2=c["name"] if len(c["name"])<=30 else c["name"][:28]+"…"
            _hdr(ax,cx2,contrib_bot+CH-0.12,CW,hdr2,col,fs=7.5)
            if c["desc"]:
                ax.text(cx2+0.22,contrib_bot+CH*0.44,_w(c["desc"],28,3),
                        ha='left',va='center',fontsize=8.5,color=MUTED,
                        zorder=5,linespacing=1.55)

        ax.plot([M+0.2,FW-M-0.2],[contrib_bot-0.04,contrib_bot-0.04],
                color=DIM,lw=0.5,alpha=0.3,zorder=3)

        out_above = contrib_bot - 0.2
    else:
        out_above = pipe_bot - 0.2

    # ── ARROW DOWN ──────────────────────────────────────────────────────────
    arrow_y = out_above
    out_band_h = 1.0
    out_y = M + 0.1
    mid_x = FW/2
    _arw_v(ax,mid_x,arrow_y,out_y+out_band_h+0.05,color=GREEN,lw=2.5)

    # ── OUTPUT BAND ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((M+0.1,out_y),FW-2*M-0.2,out_band_h,
        boxstyle="round,pad=0.1",lw=0,facecolor=GREEN,alpha=0.20,zorder=2))
    ax.add_patch(FancyBboxPatch((M+0.1,out_y),FW-2*M-0.2,out_band_h,
        boxstyle="round,pad=0.1",lw=1.5,edgecolor=GREEN+"55",facecolor='none',zorder=3))
    # Left badge
    ax.add_patch(FancyBboxPatch((M+0.2,out_y+0.28),1.6,0.44,
        boxstyle="round,pad=0.06",lw=0,facecolor=GREEN,alpha=0.90,zorder=4))
    ax.text(M+1.0,out_y+0.50,"OUTPUT",ha='center',va='center',
            fontsize=10,fontweight='bold',color=BG,zorder=5)
    out_txt=(output or "High-resolution 3D occupancy scene with semantic labels")[:140]
    ax.text(M+2.1,out_y+0.50,out_txt,ha='left',va='center',
            fontsize=8.5,color=WHITE,zorder=5)

    plt.tight_layout(pad=0)
    return fig

