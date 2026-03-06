"""
NeuraPilot Visualization Engine v5 — Full Paper Poster (Gemini-style)
"""
from __future__ import annotations
import re, textwrap
from typing import Any

BG    = "#070710"; CARD  = "#0d1117"; CARD2 = "#0f1621"
BLUE  = "#3b82f6"; AMBER = "#f59e0b"; GREEN = "#10b981"
PURP  = "#8b5cf6"; PINK  = "#ec4899"; CYAN  = "#06b6d4"
ORANGE= "#f97316"; RED   = "#ef4444"; TEAL  = "#14b8a6"
WHITE = "#f0f4f8"; MUTED = "#94a3b8"; DIM   = "#475569"
PALETTE = [BLUE, AMBER, GREEN, PURP, PINK, CYAN, ORANGE, RED, TEAL]

def _c(t):
    t = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', t)
    t = re.sub(r'\[S\d+\]|\[chunk:\d+\]', '', t)
    return re.sub(r'\s+', ' ', t).strip(' :•–—\t-')

def _w(text, w=30):
    return '\n'.join(textwrap.wrap(str(text), w)[:3])

def _s(text, n=34):
    t = _c(str(text))
    return t if len(t) <= n else t[:n-1]+'…'

# ─── ROBUST SECTION PARSER ───────────────────────────────────────────────────
def parse_sections(content: str) -> dict:
    """Parse AI explanation into structured sections."""
    result = {
        "problem": "", "datasets": [], "pipeline": [],
        "contributions": [], "losses": [], "outputs": []
    }
    
    # Split into numbered sections
    # Pattern: lines starting with digit) or digit.
    blocks = {}
    current_n = 0
    current_lines = []
    
    for line in content.splitlines():
        m = re.match(r'^(\d+)\s*[)\.]?\s+(.+)', line.strip())
        if m and int(m.group(1)) in range(1, 15):
            if current_lines:
                blocks[current_n] = '\n'.join(current_lines)
            current_n = int(m.group(1))
            current_lines = [m.group(2)]
        elif current_n > 0:
            current_lines.append(line)
    if current_lines:
        blocks[current_n] = '\n'.join(current_lines)

    for n, txt in blocks.items():
        tl = txt.lower()
        
        # PROBLEM (section 1)
        if n == 1 or any(w in tl[:60] for w in ['problem','core idea','solves','challenge']):
            # Get the substantive sentences
            sentences = re.split(r'(?<=[.!?])\s+', txt)
            result["problem"] = ' '.join(sentences[:4])[:350]
        
        # DATASETS (section 2)
        elif n == 2 or 'dataset' in tl[:40]:
            # Find dataset names: lines with A) B) or uppercase words
            for m in re.finditer(r'(?:^|\n)\s*(?:[A-Z]\)|[•\-]\s*[A-Z])\s*([A-Z][^\n]{4,60})', txt):
                ds = _c(m.group(1)).rstrip(':')
                if ds and len(ds) > 3: result["datasets"].append(ds[:55])
            # Also find patterns like "nuScenes", "KITTI", "Occ3D"
            if not result["datasets"]:
                for m in re.finditer(r'\b([A-Z][A-Za-z0-9\-]{3,30}(?:nuScenes|KITTI|Occ\w*|Scene\w*)?)\b', txt):
                    ds = m.group(1)
                    if ds not in result["datasets"] and len(ds) > 4:
                        result["datasets"].append(ds[:55])
                    if len(result["datasets"]) >= 4: break
        
        # PIPELINE (section 3)
        elif n == 3 or any(w in tl[:40] for w in ['pipeline','model','network','backbone']):
            # Numbered steps within section
            for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.{15,120})', txt):
                step = _c(m.group(1))
                if len(step) > 10: result["pipeline"].append(step[:100])
            # Bullet steps
            if not result["pipeline"]:
                for m in re.finditer(r'(?:^|\n)\s*[•\-]\s*(.{15,100})', txt):
                    result["pipeline"].append(_c(m.group(1))[:100])
        
        # CONTRIBUTIONS (sections 4,5 or "key contribution")
        elif any(w in tl[:50] for w in ['contribution','key','novel','propose']):
            # Title is the heading line
            first_line = txt.splitlines()[0] if txt.splitlines() else ''
            title = _c(first_line)[:65]
            # Clean up: remove "Key contribution #N -" prefix
            title = re.sub(r'^key\s+contribution\s*#?\d*\s*[-–—]\s*', '', title, flags=re.I)
            bullets = []
            for m in re.finditer(r'(?:^|\n)\s*[•\-]\s*(.{10,110})', txt):
                b = _c(m.group(1))
                if b and len(b) > 8: bullets.append(b[:90])
            # Also numbered sub-items
            for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.{10,110})', txt):
                b = _c(m.group(1))
                if b and b not in bullets and len(b) > 8: bullets.append(b[:90])
            if title:
                result["contributions"].append({"title": title, "bullets": bullets[:5]})
        
        # LOSSES (section 6)
        elif any(w in tl[:50] for w in ['training','loss','objective','train']):
            for m in re.finditer(r'(?:L_\w+|loss\s+\w+|\bL\w{2,8}\b)', txt):
                term = _c(m.group(0))
                if term not in result["losses"]: result["losses"].append(term[:18])
            # Also "L_occ: description" patterns
            for m in re.finditer(r'(L_\w+)\s*[:\-]\s*([^.\n]{5,80})', txt):
                lname = m.group(1)
                if lname not in result["losses"]: result["losses"].append(lname)
        
        # OUTPUTS (section 7+)
        elif n >= 7 or 'output' in tl[:30]:
            for m in re.finditer(r'(?:^|\n)\s*[•\-]\s*(.{10,100})', txt):
                result["outputs"].append(_c(m.group(1))[:90])
            # Also look for sentences describing outputs
            if not result["outputs"]:
                for sent in re.split(r'(?<=[.!?])\s+', txt):
                    sent = _c(sent)
                    if 15 < len(sent) < 120:
                        result["outputs"].append(sent[:90])
                    if len(result["outputs"]) >= 3: break

    # ── Fallbacks ────────────────────────────────────────────────────────────
    if not result["problem"]:
        # Take first substantial paragraph
        for para in content.split('\n\n'):
            para = _c(para)
            if len(para) > 80:
                result["problem"] = para[:320]; break

    if not result["datasets"]:
        result["datasets"] = ["Training Dataset A", "Benchmark Dataset B"]
    
    if not result["pipeline"]:
        # Extract numbered items anywhere
        for m in re.finditer(r'(?:^|\n)\s*\d+[.)]\s*(.{15,100})', content, re.M):
            s = _c(m.group(1))
            if s and len(s) > 10: result["pipeline"].append(s[:100])
        result["pipeline"] = result["pipeline"][:5]
    if not result["pipeline"]:
        result["pipeline"] = ["Step 1: Input Processing", "Step 2: Feature Extraction",
                               "Step 3: Transform", "Step 4: Prediction Output"]

    if not result["contributions"]:
        # Extract bold headings as contributions
        for m in re.finditer(r'\*\*([^*]{8,60})\*\*\s*[:\n]', content):
            t = _c(m.group(1))
            result["contributions"].append({"title": t, "bullets": []})
        result["contributions"] = result["contributions"][:3]
    
    if not result["losses"]:
        result["losses"] = ["L_main", "L_aux", "L_reg"]
    
    if not result["outputs"]:
        result["outputs"] = ["Final prediction output", "Semantic labels per element"]

    result["contributions"] = result["contributions"][:3]
    result["pipeline"]      = result["pipeline"][:5]
    result["outputs"]       = result["outputs"][:3]
    result["losses"]        = result["losses"][:6]

    return result


# ─── Matplotlib drawing helpers ───────────────────────────────────────────────
def _setup(w, h, dpi=130):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    return fig, ax

def _box(ax, x, y, W, H, col, fill=CARD, lw=1.8, alpha=1.0, radius=0.12):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x,y), W, H, boxstyle=f"round,pad={radius}",
        linewidth=lw, edgecolor=col, facecolor=fill, alpha=alpha, zorder=3))

def _stripe(ax, x, y, W, col, top=True, h=0.10):
    from matplotlib.patches import FancyBboxPatch
    yy = y - h if not top else y
    ax.add_patch(FancyBboxPatch((x+0.15, yy), W-0.3, h,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=col, alpha=0.85, zorder=4))

def _section_bar(ax, x, y, W, H, label, col):
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    ax.add_patch(FancyBboxPatch((x,y), W, H, boxstyle="round,pad=0.10",
        linewidth=2.0, edgecolor=col, facecolor=col+"1A", zorder=3))
    ax.add_patch(FancyBboxPatch((x, y+H-0.08), W, 0.08,
        boxstyle="round,pad=0.01", linewidth=0, facecolor=col, alpha=0.8, zorder=4))
    ax.text(x+W/2, y+H/2, label, ha="center", va="center",
            fontsize=14, fontweight="bold", color=col, zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

def _vcard(ax, x, y, W, H, title, desc, col, num=None):
    """Vertical card for pipeline steps — larger readable fonts."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch
    # Shadow
    ax.add_patch(FancyBboxPatch((x+0.07,y-0.07),W,H,boxstyle="round,pad=0.10",
        linewidth=0,facecolor="#000015",alpha=0.5,zorder=1))
    # Glow
    ax.add_patch(FancyBboxPatch((x-0.05,y-0.05),W+0.10,H+0.10,boxstyle="round,pad=0.10",
        linewidth=0,facecolor=col,alpha=0.10,zorder=2))
    # Card
    ax.add_patch(FancyBboxPatch((x,y),W,H,boxstyle="round,pad=0.10",
        linewidth=2.2,edgecolor=col,facecolor=CARD,zorder=3))
    # Top stripe
    ax.add_patch(FancyBboxPatch((x+0.15,y+H-0.15),W-0.3,0.14,boxstyle="round,pad=0.01",
        linewidth=0,facecolor=col,alpha=0.85,zorder=4))
    # Left bar
    ax.add_patch(FancyBboxPatch((x+0.06,y+H*0.15),0.12,H*0.70,boxstyle="round,pad=0.01",
        linewidth=0,facecolor=col,alpha=0.9,zorder=4))
    # Number badge
    if num is not None:
        ax.add_patch(plt.Circle((x+W*0.16,y+H-0.50),0.30,color=col,zorder=5,clip_on=False))
        ax.text(x+W*0.16,y+H-0.50,str(num),ha="center",va="center",
                fontsize=14,fontweight="bold",color=BG,zorder=6)
    # Title — bigger and more readable
    cx = x+W*0.57
    t_short = title[:24] if len(title)<=24 else title[:22]+"…"
    ax.text(cx,y+H*0.70,t_short,ha="center",va="center",fontsize=13,fontweight="bold",
            color=WHITE,path_effects=[pe.withStroke(linewidth=2,foreground=CARD)],zorder=5)
    # Desc — bigger
    if desc:
        ax.text(cx,y+H*0.30,_w(desc[:90],24),ha="center",va="center",
                fontsize=13.5,color=MUTED,zorder=5,linespacing=1.5)

def _arrow_down(ax, x, y1, y2, col=AMBER, lw=2.4):
    ax.annotate("", xy=(x,y2), xytext=(x,y1),
        arrowprops=dict(arrowstyle="->",color=col,lw=lw,mutation_scale=22),zorder=9)

def _arrow_right(ax, x1, x2, y, col=AMBER, lw=2.0):
    ax.annotate("", xy=(x2,y), xytext=(x1,y),
        arrowprops=dict(arrowstyle="->",color=col,lw=lw,mutation_scale=20),zorder=9)


# ══════════════════════════════════════════════════════════════════════════════
#  POSTER  —  full paper architecture like Gemini
# ══════════════════════════════════════════════════════════════════════════════
def draw_poster(topic: str, content: str) -> Any:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    sec      = parse_sections(content)
    problem  = sec["problem"]
    datasets = sec["datasets"]
    pipeline = sec["pipeline"]
    contribs = sec["contributions"]
    losses   = sec["losses"]
    outputs  = sec["outputs"]

    FW = 18.0
    # Compute needed height dynamically
    n_pipe  = len(pipeline)
    n_cont  = len(contribs)
    max_b   = max((len(c["bullets"]) for c in contribs), default=0)
    CH_cont = max(2.8, 0.7 + max_b*0.42)
    FH = (2.5          # header
        + 0.3 + 0.6 + 1.8   # problem
        + 0.5 + 0.6 + 1.4   # datasets
        + 0.6 + 0.6 + 2.0   # pipeline
        + 0.6 + 0.6 + CH_cont  # contributions
        + (0.8 if losses else 0)  # losses
        + 0.6 + 0.6 + 1.4   # outputs
        + 0.5)               # footer
    FH = max(FH, 22.0)

    fig, ax = _setup(FW, FH)
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)

    PAD = 0.35  # left/right margin
    IW  = FW - PAD*2   # inner width

    cy = FH  # current y — move downward

    # ── HEADER ────────────────────────────────────────────────────────────────
    cy -= 0.15
    hh = 2.1
    ax.add_patch(FancyBboxPatch((PAD, cy-hh), IW, hh,
        boxstyle="round,pad=0.12", linewidth=2.5,
        edgecolor=BLUE, facecolor="#05091c", zorder=3))
    ax.add_patch(FancyBboxPatch((PAD, cy-0.14), IW, 0.14,
        boxstyle="round,pad=0.02", linewidth=0, facecolor=BLUE, alpha=0.8, zorder=4))
    ax.text(FW/2, cy-0.85, topic[:64] if len(topic)<=64 else topic[:62]+"…",
            ha="center", va="center", fontsize=17, fontweight="bold", color=WHITE,
            path_effects=[pe.withStroke(linewidth=4,foreground="#05091c")], zorder=5)
    ax.text(FW/2, cy-1.55, "Paper Architecture  ·  Dataset → Pipeline → Contributions → Outputs",
            ha="center", va="center", fontsize=14, color=DIM, zorder=5)
    # Badge
    ax.add_patch(FancyBboxPatch((FW-4.0, cy-1.95), 3.6, 0.65,
        boxstyle="round,pad=0.08", linewidth=1.5, edgecolor=AMBER, facecolor="#1a1200", zorder=5))
    ax.text(FW-2.2, cy-1.62, "NeuraPilot Visualization", ha="center", va="center",
            fontsize=11, color=AMBER, fontweight="bold", zorder=6)
    cy -= hh + 0.3

    # ── PROBLEM ───────────────────────────────────────────────────────────────
    _section_bar(ax, PAD, cy-0.58, IW, 0.58, "THE PROBLEM / CORE IDEA", RED)
    cy -= 0.65
    ph = 1.8
    ax.add_patch(FancyBboxPatch((PAD, cy-ph), IW, ph,
        boxstyle="round,pad=0.10", linewidth=1.2, edgecolor=RED+"50", facecolor=CARD, zorder=3))
    prob_text = _c(problem)[:260]
    ax.text(FW/2, cy-ph/2, _w(prob_text, 100),
            ha="center", va="center", fontsize=14, color=MUTED, zorder=4, linespacing=1.65)
    cy -= ph + 0.35

    # ── DATASETS ──────────────────────────────────────────────────────────────
    _section_bar(ax, PAD, cy-0.58, IW, 0.58, "DATASETS", GREEN)
    cy -= 0.65
    dh = 1.3; n_ds = len(datasets)
    dw = (IW - (n_ds-1)*0.35) / max(n_ds, 1)
    for i, ds in enumerate(datasets):
        dx = PAD + i*(dw+0.35)
        col = PALETTE[(i+2) % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((dx, cy-dh), dw, dh,
            boxstyle="round,pad=0.10", linewidth=1.8,
            edgecolor=col, facecolor="#060e08", zorder=3))
        ax.add_patch(FancyBboxPatch((dx+0.1, cy-0.12), dw-0.2, 0.10,
            boxstyle="round,pad=0.01", linewidth=0, facecolor=col, alpha=0.8, zorder=4))
        ax.text(dx+dw/2, cy-dh/2, _s(ds, 32),
                ha="center", va="center", fontsize=13, fontweight="bold", color=WHITE, zorder=5)
    cy -= dh + 0.25

    # Arrow: datasets → pipeline
    _arrow_down(ax, FW/2, cy, cy-0.55)
    cy -= 0.60

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    _section_bar(ax, PAD, cy-0.58, IW, 0.58, "MODEL PIPELINE", BLUE)
    cy -= 0.65
    n_p = len(pipeline)
    pw  = (IW - (n_p-1)*0.45) / max(n_p, 1)
    ph2 = 2.0
    for i, step in enumerate(pipeline):
        px = PAD + i*(pw+0.45)
        col = PALETTE[i % len(PALETTE)]
        # Extract short title (first chunk before colon or first 5 words)
        parts = re.split(r'[:\-–]', step, 1)
        title = _c(parts[0])[:28]
        desc  = _c(parts[1]) if len(parts) > 1 else ""
        if not title.strip():
            words = step.split()
            title = ' '.join(words[:4])
            desc  = ' '.join(words[4:])
        _vcard(ax, px, cy-ph2, pw, ph2, title, desc, col, num=i+1)
        if i < n_p-1:
            _arrow_right(ax, px+pw+0.02, px+pw+0.43, cy-ph2+ph2/2)
    cy -= ph2 + 0.30

    # Arrow: pipeline → contributions
    _arrow_down(ax, FW/2, cy, cy-0.55, col=PURP)
    cy -= 0.60

    # ── CONTRIBUTIONS ─────────────────────────────────────────────────────────
    if contribs:
        _section_bar(ax, PAD, cy-0.58, IW, 0.58, "KEY CONTRIBUTIONS", PURP)
        cy -= 0.65
        n_c = len(contribs)
        cw  = (IW - (n_c-1)*0.40) / max(n_c, 1)

        for i, contrib in enumerate(contribs):
            cx2 = PAD + i*(cw+0.40)
            col = PALETTE[(i+3) % len(PALETTE)]
            bullets = contrib["bullets"]
            ch2 = max(2.8, 0.75 + len(bullets)*0.44)
            # Shadow + glow
            ax.add_patch(FancyBboxPatch((cx2+0.07, cy-ch2-0.07), cw, ch2,
                boxstyle="round,pad=0.10", linewidth=0, facecolor="#000015", alpha=0.5, zorder=1))
            ax.add_patch(FancyBboxPatch((cx2-0.05, cy-ch2-0.05), cw+0.10, ch2+0.10,
                boxstyle="round,pad=0.10", linewidth=0, facecolor=col, alpha=0.09, zorder=2))
            ax.add_patch(FancyBboxPatch((cx2, cy-ch2), cw, ch2,
                boxstyle="round,pad=0.10", linewidth=2.0, edgecolor=col, facecolor=CARD, zorder=3))
            ax.add_patch(FancyBboxPatch((cx2+0.10, cy-0.14), cw-0.2, 0.12,
                boxstyle="round,pad=0.01", linewidth=0, facecolor=col, alpha=0.85, zorder=4))
            ax.add_patch(FancyBboxPatch((cx2+0.06, cy-ch2*0.90), 0.10, ch2*0.80,
                boxstyle="round,pad=0.01", linewidth=0, facecolor=col, alpha=0.85, zorder=4))

            # Title — wrapped
            title_wrapped = textwrap.wrap(contrib["title"][:70], int(cw*4.2))
            ty = cy - 0.50
            for tl in title_wrapped[:2]:
                ax.text(cx2+cw/2+0.06, ty, tl,
                        ha="center", va="center", fontsize=13, fontweight="bold", color=WHITE, zorder=5)
                ty -= 0.30

            # Bullets
            for ji, bullet in enumerate(bullets[:5]):
                by2 = cy - 0.52 - len(title_wrapped[:2])*0.30 - 0.20 - ji*0.42
                # Bullet dot
                import matplotlib.patches as mp
                ax.add_patch(mp.FancyBboxPatch((cx2+0.22, by2-0.07), 0.11, 0.15,
                    boxstyle="round,pad=0.01", linewidth=0, facecolor=col, alpha=0.9, zorder=4))
                ax.text(cx2+0.40, by2+0.003, _c(bullet)[:58],
                        ha="left", va="center", fontsize=11, color=MUTED, zorder=5)

        cy -= max(2.8, 0.75 + max((len(c["bullets"]) for c in contribs), default=0)*0.44) + 0.25

    # ── LOSSES ────────────────────────────────────────────────────────────────
    if losses:
        _arrow_down(ax, FW/2, cy, cy-0.40, col=AMBER)
        cy -= 0.45
        ax.text(FW/2, cy-0.24, "Training Objectives", ha="center", va="center",
                fontsize=13, color=MUTED, fontweight="600", zorder=5)
        cy -= 0.38
        n_l = len(losses); lw2 = (IW - (n_l-1)*0.25) / max(n_l, 1)
        for i, loss in enumerate(losses):
            lx = PAD + i*(lw2+0.25); col = PALETTE[i%len(PALETTE)]
            ax.add_patch(FancyBboxPatch((lx, cy-0.52), lw2, 0.52,
                boxstyle="round,pad=0.08", linewidth=1.8,
                edgecolor=col, facecolor=CARD, zorder=3))
            ax.text(lx+lw2/2, cy-0.26, loss[:18],
                    ha="center", va="center", fontsize=14, fontweight="bold", color=col, zorder=5)
        cy -= 0.60

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    _arrow_down(ax, FW/2, cy, cy-0.45, col=GREEN)
    cy -= 0.50
    _section_bar(ax, PAD, cy-0.58, IW, 0.58, "OUTPUTS", GREEN)
    cy -= 0.65
    n_o = len(outputs)
    ow  = (IW - (n_o-1)*0.35) / max(n_o, 1); oh = 1.2
    for i, out in enumerate(outputs):
        ox = PAD + i*(ow+0.35)
        ax.add_patch(FancyBboxPatch((ox, cy-oh), ow, oh,
            boxstyle="round,pad=0.10", linewidth=2.0, edgecolor=GREEN, facecolor="#060e08", zorder=3))
        ax.add_patch(FancyBboxPatch((ox+0.10, cy-0.12), ow-0.2, 0.10,
            boxstyle="round,pad=0.01", linewidth=0, facecolor=GREEN, alpha=0.8, zorder=4))
        ax.text(ox+ow/2, cy-oh/2, _w(_c(out)[:90], int(ow*3.2)),
                ha="center", va="center", fontsize=11.5, color=WHITE, zorder=5, linespacing=1.4)
    cy -= oh + 0.3

    # ── FOOTER ────────────────────────────────────────────────────────────────
    if cy > 0.4:
        ax.text(FW/2, max(0.2, cy/2), "Generated by NeuraPilot  ·  Visualization Engine v5",
                ha="center", va="center", fontsize=13, color=DIM, style="italic", zorder=5)

    plt.tight_layout(pad=0.0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  OTHER DIAGRAM TYPES (pipeline, mindmap, comparison, timeline, architecture)
# ══════════════════════════════════════════════════════════════════════════════
def extract_concepts(content):
    pairs, seen = [], set()
    def _add(t, d):
        t, d = _c(t), _c(d)
        k = t.lower()[:24]
        if len(t) > 2 and k not in seen:
            pairs.append((t, d)); seen.add(k)
    for m in re.finditer(r'\*\*([^*]{3,60})\*\*\s*[:\-\u2013]\s*([^*\n]{10,300})', content):
        _add(m.group(1), m.group(2)[:160])
    if len(pairs) < 3:
        for line in content.splitlines():
            line = line.strip()
            if not line or line[0] not in '•*-–1234567890': continue
            c2 = re.sub(r'^[•*\-–\d\.\)]+\s*', '', line)
            c2 = re.sub(r'\*\*([^*]+)\*\*', r'\1', c2)
            pts = re.split(r'[:\-\u2013]\s', c2, 1)
            if len(pts) == 2 and 3 < len(pts[0]) < 50 and len(pts[1]) > 8:
                _add(pts[0], pts[1][:160])
    if len(pairs) < 3:
        for s in re.split(r'(?<=[.!?])\s+', content):
            s = _c(s)
            if 20 < len(s) < 200:
                w2 = s.split()
                if len(w2) >= 4: _add(' '.join(w2[:3]), ' '.join(w2[3:40]))
            if len(pairs) >= 6: break
    return pairs[:8] or [("Concept A","Key idea"),("Concept B","Key idea"),("Concept C","Key idea")]

def pick_dtype(query, content, override="auto"):
    if override and override.lower() not in ("auto","auto (best fit)"):
        m = override.lower()
        if "pipeline" in m or "flow" in m: return "pipeline"
        if "architecture" in m: return "architecture"
        if "concept" in m or "map" in m or "mind" in m: return "mindmap"
        if "comparison" in m or "side" in m: return "comparison"
        if "timeline" in m: return "timeline"
        if "poster" in m or "paper" in m or "overview" in m: return "poster"
    low = content.lower()
    has_sections = len(re.findall(r'^\d+[)\.]', content, re.M)) >= 3
    has_paper_keywords = sum(1 for w in ["dataset","pipeline","output","contribution","training","model"] if w in low)
    if has_sections and has_paper_keywords >= 3: return "poster"
    if any(w in low for w in ["compare","versus"," vs ","differ","unlike"]): return "comparison"
    if any(w in low for w in ["step","stage","process","algorithm","pipeline","how to"]): return "pipeline"
    if any(w in low for w in ["timeline","history","evolution","over time"]): return "timeline"
    if any(w in low for w in ["architecture","system","module","layer","component"]): return "architecture"
    return "mindmap"

def draw_pipeline(topic, concepts, fig_w=16):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    n=len(concepts); COLS=3; CW,CH,GX,GY=4.2,2.8,0.75,1.4
    rows=(n+COLS-1)//COLS; total_w=COLS*CW+(COLS-1)*GX; sx=(fig_w-total_w)/2
    fig_h=2.1+rows*(CH+GY)
    fig,ax=_setup(fig_w,fig_h); ax.set_xlim(0,fig_w); ax.set_ylim(0,fig_h)
    ax.text(fig_w/2,fig_h-0.32,topic,ha="center",va="top",fontsize=19,fontweight="bold",
            color=WHITE,path_effects=[pe.withStroke(linewidth=5,foreground=BG)],zorder=10)
    ax.text(fig_w/2,fig_h-0.85,"Step-by-Step Process Flow",ha="center",va="top",fontsize=13.5,color=DIM,zorder=10)
    ax.plot([fig_w*0.06,fig_w*0.94],[fig_h-1.12,fig_h-1.12],color=BLUE,lw=0.5,alpha=0.30,zorder=10)
    pos={}
    for i,(t,d) in enumerate(concepts):
        r,c2=divmod(i,COLS); x=sx+c2*(CW+GX); y=(fig_h-1.8)-r*(CH+GY)-CH
        _vcard(ax,x,y,CW,CH,t,d,PALETTE[i%len(PALETTE)],num=i+1); pos[i]=(x,y,CW,CH)
    for i in range(n-1):
        x1,y1,w1,h1=pos[i]; x2,y2,w2,h2=pos[i+1]; r1,r2=i//COLS,(i+1)//COLS
        if r1==r2:
            ax.annotate("",xy=(x2,y2+h2/2),xytext=(x1+w1,y1+h1/2),
                arrowprops=dict(arrowstyle="->",color=AMBER,lw=2.2,mutation_scale=22),zorder=8)
        else:
            ax.annotate("",xy=(x2+w2/2,y2+h2),xytext=(x1+w1/2,y1),
                arrowprops=dict(arrowstyle="->",color=AMBER,lw=2.2,mutation_scale=22,
                                connectionstyle="arc3,rad=0.0"),zorder=8)
    plt.tight_layout(pad=0.3); return fig

def draw_mindmap(topic, concepts):
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    n=len(concepts)
    fig,ax=_setup(15,11); ax.set_xlim(-7.5,7.5); ax.set_ylim(-5.5,5.5)
    CW,CH=6.0,2.0
    ax.add_patch(FancyBboxPatch((-CW/2,-CH/2),CW,CH,boxstyle="round,pad=0.18",linewidth=3.5,edgecolor=AMBER,facecolor="#140f00",zorder=8))
    ax.add_patch(FancyBboxPatch((-CW/2,-CH/2),CW,CH,boxstyle="round,pad=0.18",linewidth=0,facecolor=AMBER,alpha=0.09,zorder=7))
    short=topic if len(topic)<=34 else topic[:32]+"…"
    ax.text(0,0.22,short,ha="center",va="center",fontsize=15,fontweight="bold",color=AMBER,
            path_effects=[pe.withStroke(linewidth=4,foreground="#140f00")],zorder=9)
    ax.text(0,-0.52,"Core Concept",ha="center",va="center",fontsize=14,color="#b8860b",zorder=9)
    angles=np.linspace(np.pi/2,np.pi/2+2*np.pi,n,endpoint=False)
    radii=[4.2 if i%2==0 else 4.6 for i in range(n)]
    NW,NH=3.4,1.35
    for i,((title,desc),angle,R) in enumerate(zip(concepts,angles,radii)):
        color=PALETTE[i%len(PALETTE)]; nx=R*np.cos(angle); ny=R*np.sin(angle)
        sx0=1.8*np.cos(angle); sy0=1.8*np.sin(angle)
        ex=(R-1.9)*np.cos(angle); ey=(R-1.9)*np.sin(angle)
        ax.plot([sx0,ex],[sy0,ey],color=color,lw=1.8,alpha=0.35,zorder=2)
        ax.annotate("",xy=(nx-1.8*np.cos(angle),ny-1.8*np.sin(angle)),xytext=(ex,ey),
            arrowprops=dict(arrowstyle="->",color=color,lw=1.8,mutation_scale=16),zorder=3)
        bx,by=nx-NW/2,ny-NH/2
        ax.add_patch(FancyBboxPatch((bx-0.07,by-0.07),NW+0.14,NH+0.14,boxstyle="round,pad=0.10",linewidth=0,facecolor=color,alpha=0.10,zorder=4))
        ax.add_patch(FancyBboxPatch((bx,by),NW,NH,boxstyle="round,pad=0.10",linewidth=2.0,edgecolor=color,facecolor=CARD,zorder=5))
        ax.add_patch(FancyBboxPatch((bx,by+NH-0.10),NW,0.10,boxstyle="round,pad=0.02",linewidth=0,facecolor=color,alpha=0.80,zorder=6))
        dn=title if len(title)<=28 else title[:26]+"…"
        ax.text(nx,by+NH*0.65,dn,ha="center",va="center",fontsize=13,fontweight="bold",color=WHITE,zorder=7)
        if desc: ax.text(nx,by+NH*0.22,desc[:42]+("…" if len(desc)>42 else ""),ha="center",va="center",fontsize=11,color=MUTED,zorder=7)
    ax.set_title(topic,pad=18,fontsize=19,fontweight="bold",color=WHITE,path_effects=[pe.withStroke(linewidth=5,foreground=BG)])
    plt.tight_layout(pad=0.2); return fig

def draw_comparison(topic, concepts):
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    half=max(len(concepts)//2,2); left=concepts[:half]; right=concepts[half:half*2] if len(concepts)>=half*2 else concepts[:half]
    rows=max(len(left),len(right)); fig_h=max(9.0,rows*1.55+3.5)
    fig,(axL,axR)=plt.subplots(1,2,figsize=(16,fig_h)); fig.patch.set_facecolor(BG)
    fig.suptitle(topic,fontsize=18,fontweight="bold",color=WHITE,y=0.98,path_effects=[pe.withStroke(linewidth=5,foreground=BG)])
    def _side(ax,nodes,col,letter):
        ax.set_facecolor(BG); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
        head=nodes[0][0][:26] if nodes else ""
        ax.add_patch(FancyBboxPatch((0.15,8.05),9.7,1.75,boxstyle="round,pad=0.12",linewidth=2.5,edgecolor=col,facecolor=col+"25",zorder=2))
        ax.add_patch(FancyBboxPatch((0.15,9.70),9.7,0.12,boxstyle="round,pad=0.02",linewidth=0,facecolor=col,alpha=0.9,zorder=3))
        ax.text(5,8.98,f"[{letter}]  {head}",ha="center",va="center",fontsize=14,fontweight="bold",color=WHITE,zorder=4)
        items=nodes[1:] if len(nodes)>1 else nodes
        row_h=min(1.55,(8.0-0.3)/max(len(items),1))
        for ji,(ft,fd) in enumerate(items[:6]):
            yp=7.6-ji*row_h
            ax.add_patch(FancyBboxPatch((0.15,yp-row_h*0.44),9.7,row_h*0.88,boxstyle="round,pad=0.08",linewidth=1.3,edgecolor=col+"45",facecolor=CARD2,zorder=2))
            ax.add_patch(FancyBboxPatch((0.15,yp+row_h*0.42),9.7,0.05,boxstyle="round,pad=0.01",linewidth=0,facecolor=col,alpha=0.4,zorder=3))
            ax.text(0.65,yp,"*",ha="left",va="center",fontsize=14,color=col,fontweight="bold",zorder=4)
            ax.text(1.15,yp+0.17,ft[:30],ha="left",va="center",fontsize=13,fontweight="bold",color=WHITE,zorder=4)
            if fd: ax.text(1.15,yp-0.22,fd[:65]+("…" if len(fd)>65 else ""),ha="left",va="center",fontsize=11,color=MUTED,zorder=4)
    _side(axL,left,BLUE,"A"); _side(axR,right,AMBER,"B")
    plt.tight_layout(rect=[0,0,1,0.96]); return fig

def draw_timeline(topic, concepts):
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    n=len(concepts)
    fig,ax=_setup(16,7.5); ax.set_xlim(0,16); ax.set_ylim(0,7.5)
    ax.text(8,7.18,topic,ha="center",va="top",fontsize=19,fontweight="bold",color=WHITE,
            path_effects=[pe.withStroke(linewidth=5,foreground=BG)],zorder=10)
    ax.plot([0.8,15.2],[3.75,3.75],color=AMBER,lw=3.0,alpha=0.30,zorder=2)
    ax.annotate("",xy=(15.6,3.75),xytext=(14.8,3.75),
        arrowprops=dict(arrowstyle="->",color=AMBER,lw=2.2,mutation_scale=22),zorder=9)
    step=14.0/max(n-1,1)
    for i,(title,desc) in enumerate(concepts):
        x=0.8+i*step; color=PALETTE[i%len(PALETTE)]; above=(i%2==0)
        ax.scatter([x],[3.75],s=260,color=color,zorder=6,edgecolors=BG,linewidths=3)
        ax.text(x,3.75,str(i+1),ha="center",va="center",fontsize=14,fontweight="bold",color=BG,zorder=7)
        by=5.20 if above else 1.50; BW=min(3.0,step*0.86); BH=1.40
        ax.add_patch(FancyBboxPatch((x-BW/2-0.06,by-BH/2-0.06),BW+0.12,BH+0.12,boxstyle="round,pad=0.10",linewidth=0,facecolor=color,alpha=0.10,zorder=3))
        ax.add_patch(FancyBboxPatch((x-BW/2,by-BH/2),BW,BH,boxstyle="round,pad=0.10",linewidth=1.8,edgecolor=color,facecolor=CARD,zorder=4))
        ax.add_patch(FancyBboxPatch((x-BW/2,by+BH/2-0.09),BW,0.09,boxstyle="round,pad=0.02",linewidth=0,facecolor=color,alpha=0.85,zorder=5))
        ax.text(x,by+0.14,title[:22],ha="center",va="center",fontsize=13,fontweight="bold",color=WHITE,zorder=6)
        if desc: ax.text(x,by-0.32,desc[:30],ha="center",va="center",fontsize=11,color=MUTED,zorder=6)
        c1=3.88 if above else 3.62; c2=by-BH/2 if above else by+BH/2
        ax.plot([x,x],[c1,c2],color=color,lw=1.5,alpha=0.5,zorder=3)
    plt.tight_layout(pad=0.2); return fig

def draw_architecture(topic, concepts):
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    n=len(concepts); CW,CH=3.2,2.1
    total_w=n*CW+(n-1)*0.55; fig_w=max(16,total_w+2.0); fig_h=10.0
    fig,ax=_setup(fig_w,fig_h); ax.set_xlim(0,fig_w); ax.set_ylim(0,fig_h)
    ax.text(fig_w/2,fig_h-0.32,topic,ha="center",va="top",fontsize=19,fontweight="bold",
            color=WHITE,path_effects=[pe.withStroke(linewidth=5,foreground=BG)],zorder=10)
    top_x=(fig_w-7.0)/2
    _vcard(ax,top_x,fig_h-3.9,7.0,2.0,topic[:24] if len(topic)<=24 else topic[:22]+"…","Core System",AMBER)
    mid_x=fig_w/2
    ax.plot([mid_x,mid_x],[fig_h-3.9,fig_h-4.7],color=AMBER,lw=2.5,alpha=0.6,zorder=5)
    bus_y=fig_h-5.0; csx=(fig_w-total_w)/2
    ax.plot([csx+CW*0.5,csx+total_w-CW*0.5],[bus_y,bus_y],color=BLUE,lw=2.0,alpha=0.40,zorder=4)
    comp_y=bus_y-CH-0.6
    for i,(name,desc) in enumerate(concepts):
        cx=csx+i*(CW+0.55); cx_mid=cx+CW/2; color=PALETTE[i%len(PALETTE)]
        _vcard(ax,cx,comp_y,CW,CH,name,desc,color)
        ax.plot([cx_mid,cx_mid],[bus_y,comp_y+CH],color=color,lw=1.8,alpha=0.55,zorder=5)
        ax.scatter([cx_mid],[bus_y],s=70,color=color,zorder=6,edgecolors=BG,linewidths=2)
    plt.tight_layout(pad=0.3); return fig

# ── Master entry point ────────────────────────────────────────────────────────
def render_diagram(topic, content, diagram_type_override="auto"):
    """Returns (fig, dtype, concepts)"""
    concepts = extract_concepts(content)
    dtype    = pick_dtype(topic, content, diagram_type_override)
    if dtype == "poster":
        fig = draw_poster(topic, content)
    elif dtype == "pipeline":
        fig = draw_pipeline(topic, concepts)
    elif dtype == "comparison":
        fig = draw_comparison(topic, concepts)
    elif dtype == "timeline":
        fig = draw_timeline(topic, concepts)
    elif dtype == "architecture":
        fig = draw_architecture(topic, concepts)
    else:
        fig = draw_mindmap(topic, concepts)
        dtype = "mindmap"
    return fig, dtype, concepts
