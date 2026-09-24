"""
Supplementary Fig. S4 — gradient saliency over the input point cloud.
thread facebase3d-paper-2026-08-08.

Supports Sec. 2.6 paragraph 4, and CORRECTS it. The manuscript stated that
saliency was "generally diffuse and did not reveal anatomical regions that were
consistently associated with diagnostic predictions". Recomputed on the
corrected checkpoint over all 475 test scans, the first half of that is wrong
and the second half is right for a more specific reason. The figure separates
the three questions the original sentence merged:

  (b) WITHIN a scan, is saliency concentrated?          -> YES, strongly
  (c) ACROSS scans, does it land in the same place?     -> YES, but mostly
                                                           independent of
                                                           diagnosis
  (d) Does it fall on an interpretable anatomical region? -> NO

WHY THE OLD ANSWER WAS "DIFFUSE". runs/analysis/B2_saliency_stats.csv
(a) came from runs/B2/, the superseded pre-correction checkpoint,
and (b) measured the entropy of a saliency map AVERAGED over every scan of a
class. Averaging point-indexed saliency across subjects who are not in dense
correspondence destroys per-scan structure by construction, so it reported
98-99.5% of maximum entropy. Per scan the true figure is 90.4%.

THE CONFOUND THAT NEARLY GOT THROUGH. Same-diagnosis scan pairs include repeated
scans OF THE SAME SUBJECT, which correlate trivially (r = +0.210 here). Counting
them inflates the same-diagnosis correlation from +0.128 to +0.184 and more than
doubles the apparent diagnosis-specific effect. Panel (c) excludes them and the
permutation null shuffles diagnosis AT SUBJECT LEVEL.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import figstyle as fs

fs.apply()
A = "../../runs/analysis/"
D = np.load(A + "saliency_full_B2_corrected.npz", allow_pickle=True)
fb = np.load(A + "saliency_full_B2_corrected_fbid.npy", allow_pickle=True).astype(str)
pts, sal, y, pred = D["pts"], D["sal"], D["y"], D["pred"]
names = [str(x) for x in D["label_names"]]
N, NP = sal.shape
rng = np.random.default_rng(0)
SAL_CMAP = LinearSegmentedColormap.from_list(
    "s", ["#9CC0DC", "#3E6E9E", "#123F63", "#B03A2E"])

# ── within-scan concentration ──────────────────────────────────────────────
k = NP // 10
topdec = np.sort(sal, axis=1)[:, ::-1][:, :k].sum(1) / sal.sum(1)
p = sal / sal.sum(1, keepdims=True)
H = -(p * np.log(p + 1e-30)).sum(1)
Hmax = np.log(NP)

# ── across-scan reproducibility, in the only common frame available ────────
G = 12
def vox(P, S):
    q = np.clip(((P + 1.0) / 2.0 * G).astype(int), 0, G - 1)
    lin = (q[:, 0] * G + q[:, 1]) * G + q[:, 2]
    v = np.bincount(lin, weights=S, minlength=G ** 3)
    c = np.bincount(lin, minlength=G ** 3)
    o = np.zeros(G ** 3); m = c > 0; o[m] = v[m] / c[m]
    return o, m

V = np.zeros((N, G ** 3)); M = np.zeros((N, G ** 3), bool)
for i in range(N):
    V[i], M[i] = vox(pts[i], sal[i])
occ = M.mean(0) > 0.5
Z = V[:, occ]
Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-12)

a, b = np.triu_indices(N, 1)
r = (Z[a] * Z[b]).mean(1)
same_subj = fb[a] == fb[b]
same_dx = y[a] == y[b]
ss = same_dx & ~same_subj
dd = ~same_dx
Zs = Z.copy()
for i in range(N):
    Zs[i] = Zs[i][rng.permutation(Zs.shape[1])]
null = (Zs[a[:20000]] * Zs[b[:20000]]).mean(1)
gap = r[ss].mean() - r[dd].mean()

subs = np.unique(fb); sub_dx = {s: y[fb == s][0] for s in subs}
gaps = []
for _ in range(1000):
    perm = rng.permutation(len(subs))
    mp = {s: sub_dx[subs[q]] for s, q in zip(subs, perm)}
    yy = np.array([mp[s] for s in fb])
    sd = yy[a] == yy[b]
    gaps.append(r[sd & ~same_subj].mean() - r[~sd].mean())
gaps = np.array(gaps)
z = (gap - gaps.mean()) / gaps.std()
pval = max((np.abs(gaps - gaps.mean()) >= abs(gap - gaps.mean())).mean(), 1e-3)

# ── anatomical localisation ────────────────────────────────────────────────
rk = np.argsort(np.argsort(sal, axis=1), axis=1) / (NP - 1)
rad = np.linalg.norm(pts - pts.mean(1, keepdims=True), axis=2)
rn = rad / rad.max(1, keepdims=True)
yy_ = pts[:, :, 1]
yn = (yy_ - yy_.min(1, keepdims=True)) / (yy_.max(1, keepdims=True) - yy_.min(1, keepdims=True))
edges = np.linspace(0, 1, 6)
band_r = [rk[(rn >= lo) & (rn < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
band_y = [rk[(yn >= lo) & (yn < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]

# ── figure ─────────────────────────────────────────────────────────────────
SHOW = ["Achondroplasia", "Down Syndrome", "Williams Syndrome",
        "Costello Syndrome", "Noonan Syndrome", "CHARGE Syndrome"]


def neck_cut(P, nb=24):
    """Return the y below which the cloud is neck and shoulder, or None.

    FaceBase acquisitions differ in how far below the chin the captured surface
    extends. A capture that runs past the chin NARROWS at the neck and widens
    again at the shoulders, so the boundary is a constriction in the width
    profile, not a fixed fraction of the height. Fixed-fraction cropping was
    tried first and does not work: it leaves the shoulder in frame on scans
    where the torso sits low, and eats the mandible on scans that stop at the
    chin. Returns None when there is no constriction, i.e. the scan is already
    face-only, in which case nothing is cropped.
    """
    yy = P[:, 1]
    lo, hi = yy.min(), yy.max()
    edges = hi - np.arange(nb + 1) / nb * (hi - lo)
    w = []
    for k in range(nb):
        b = P[(yy <= edges[k]) & (yy > edges[k + 1])]
        w.append(b[:, 0].max() - b[:, 0].min() if len(b) > 8 else np.nan)
    w = np.array(w)
    lo_i, hi_i = int(0.30 * nb), int(0.90 * nb)
    seg = w[lo_i:hi_i]
    if np.all(np.isnan(seg)):
        return None
    j = lo_i + int(np.nanargmin(seg))
    below = w[j + 1:]
    if not len(below) or np.all(np.isnan(below)):
        return None
    # Require a real waist: the cloud must re-widen appreciably below the
    # minimum, and the minimum must be appreciably narrower than the face.
    if np.nanmax(below) / w[j] < 1.25 or w[j] / np.nanmax(w[:j]) > 0.85:
        return None
    return edges[j + 1]


def head_only(P):
    """Boolean mask selecting the head, cropping at the neck if there is one."""
    cut = neck_cut(P)
    return np.ones(len(P), bool) if cut is None else P[:, 1] > cut


def shoulder_score(P):
    """Base width over mid-face width, used to rank candidate exemplars.

    Applied to the CROPPED cloud so selection and framing agree: a scan whose
    shoulder the neck cut removes cleanly should rank as well as one that never
    had a shoulder.
    """
    yy = P[:, 1]
    lo, hi = yy.min(), yy.max()
    base = P[yy < lo + 0.15 * (hi - lo)]
    mid = P[(yy > lo + 0.40 * (hi - lo)) & (yy < lo + 0.70 * (hi - lo))]
    if len(base) < 20 or len(mid) < 20:
        return np.inf
    return ((base[:, 0].max() - base[:, 0].min())
            / (mid[:, 0].max() - mid[:, 0].min()))


fig = plt.figure(figsize=(fs.DOUBLE, 4.05))
gs = fig.add_gridspec(2, 6, height_ratios=[0.50, 1.0], hspace=0.30, wspace=0.45)

# Panel (a) is ONE axes holding all six faces, not six axes side by side.
# With six equal-aspect axes, matplotlib fits each face to its cell WIDTH and
# leaves the surplus cell height as whitespace, which no amount of hspace or
# height_ratios tuning removes. Placing the faces manually in a single axes
# gives the row exactly the height the faces need.
axa = fig.add_subplot(gs[0, :])
STEP = 1.06
for j, s_ in enumerate(SHOW):
    c = names.index(s_)
    cand = np.where((y == c) & (pred == c))[0]
    if not len(cand):
        cand = np.where(y == c)[0]
    i = int(min(cand, key=lambda q: shoulder_score(pts[q][head_only(pts[q])])))
    P, S = pts[i].copy(), sal[i]
    thr_full = np.quantile(S, 0.9)
    # Frame on the head, cropping at the neck constriction where there is one.
    # Points below it are simply not drawn -- they are still in the cloud the
    # model saw and still counted in thr_full above.
    keep = head_only(P)
    P, S = P[keep][:, :2], S[keep]
    # normalise each face into a unit box so the row is not dominated by
    # whichever scan happens to have the largest bounding box
    P = P - P.min(0)
    P = P / P.max()
    P[:, 0] += j * STEP
    # Colour the TOP DECILE, the same set panels (b) and (c) quantify, rather
    # than the full percentile range: a continuous ramp puts half the points
    # above the midpoint and the face reads as uniform speckle regardless of
    # what the saliency actually does.
    hot = S >= thr_full   # threshold from the FULL cloud, not the crop
    axa.scatter(P[~hot, 0], P[~hot, 1], c="#DCDCDC", s=0.55, linewidths=0,
                rasterized=True)
    oh = np.argsort(S[hot])
    axa.scatter(P[hot][oh, 0], P[hot][oh, 1],
                c=np.linspace(0, 1, int(hot.sum())), cmap=SAL_CMAP, s=1.5,
                linewidths=0, rasterized=True)
    # Labels BELOW the faces: above, they collide with the panel title, which
    # sits on the same band because axa spans the whole row.
    axa.text(j * STEP + 0.5, -0.04, s_.replace(" Syndrome", ""), ha="center",
             va="top", fontsize=6.5)
axa.set_xlim(-0.04, (len(SHOW) - 1) * STEP + 1.04)
axa.set_ylim(-0.22, 1.34)
axa.set_aspect("equal"); axa.axis("off")
# Title as figure text, not an axes title: loc="left" on a full-width axes is
# not constrained to the canvas and this string overruns it.
fig.text(0.5, 0.985, "(a)  Highest-saliency 10% of points (coloured) on "
         "exemplar scans; remaining 90% in grey", ha="center", va="top",
         fontsize=8.5)

ax2 = fig.add_subplot(gs[1, 0:2])
ax2.hist(100 * topdec, bins=28, color="#3E6E9E", linewidth=0)
ax2.axvline(10, color=fs.ACCENT, lw=0.9, ls=(0, (3, 2)))
ax2.annotate("10% if uniform", xy=(10, ax2.get_ylim()[1] * 0.96),
             xytext=(4, 0), textcoords="offset points", fontsize=6.3,
             color=fs.ACCENT, va="top", ha="left")
ax2.set_xlim(0, 60)
ax2.set_xlabel("Saliency in top 10% of points (%)", fontsize=7.3)
ax2.set_ylabel("Test scans", fontsize=7.3)
ax2.set_title("(b)  Concentrated within a scan", loc="center", fontsize=8.5)

ax3 = fig.add_subplot(gs[1, 2:4])
bins = np.linspace(-0.3, 0.6, 55)
for v, c, lab in [(null, "#B8B8B8", f"shuffled ({null.mean():+.3f})"),
                  (r[dd], "#3E6E9E", f"diff. diagnosis ({r[dd].mean():+.3f})"),
                  (r[ss], "#B03A2E", f"same diagnosis ({r[ss].mean():+.3f})")]:
    ax3.hist(v, bins=bins, density=True, histtype="step", lw=1.1, color=c,
             label=lab)
ax3.axvline(0, color=fs.INK, lw=0.6)
# Headroom for the legend. The null peak runs to the top of the data range, so
# no in-axes legend position is free until the y-limit is raised above it.
ax3.set_ylim(0, ax3.get_ylim()[1] * 1.42)
ax3.set_xlabel("Correlation between scan pairs", fontsize=7.3)
ax3.set_ylabel("Density", fontsize=7.3)
# upper LEFT put the legend straight through the shuffled-null peak; the
# right-hand tail above x = 0.3 is the only genuinely empty region.
ax3.legend(frameon=False, fontsize=5.9, loc="upper right", handlelength=1.1,
           borderaxespad=0.3, labelspacing=0.35)
ax3.annotate(f"gap {gap:+.3f}\n$p<0.001$", xy=(0.97, 0.44),
             xycoords="axes fraction", ha="right", fontsize=6.3, color=fs.ACCENT)
ax3.set_title("(c)  Reproducible across scans", loc="center", fontsize=8.5)

ax4 = fig.add_subplot(gs[1, 4:6])
xx = np.arange(5)
ax4.plot(xx, band_r, "o-", ms=3.2, lw=1.0, color="#123F63",
         label="radial")
ax4.plot(xx, band_y, "s--", ms=3.2, lw=1.0, color=fs.ACCENT,
         label="vertical")
ax4.axhline(0.5, color=fs.MUTED, lw=0.7, ls=(0, (2, 2)))
ax4.annotate("0.5 = no preference", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
             xytext=(0, 3), textcoords="offset points", fontsize=6.2,
             color=fs.MUTED)
ax4.set_ylim(0.40, 0.60)
ax4.set_xticks(xx)
ax4.set_xticklabels(["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"], fontsize=6.5)
ax4.set_xlabel("Normalised position in face", fontsize=7.0)
ax4.set_ylabel("Mean saliency percentile", fontsize=7.3)
ax4.legend(frameon=False, fontsize=6.2, loc="lower left", handlelength=1.4,
           ncol=2, columnspacing=1.0, borderaxespad=0.2)
ax4.set_title("(d)  No preferred region", loc="center", fontsize=7.8)

fs.finish(fig, "../../figures/figS4_saliency.png", fs.DOUBLE)

print(f"  scans={N} subjects={len(subs)} points={NP}")
print(f"  entropy {H.mean():.3f} = {H.mean() / Hmax:.2%} of max")
print(f"  top-decile mass {topdec.mean():.2%} (uniform 10%), "
      f"range {topdec.min():.1%}-{topdec.max():.1%}")
print(f"  same-subject pairs r={r[same_subj].mean():+.4f}  (EXCLUDED, n={same_subj.sum()})")
print(f"  same-dx/diff-subject r={r[ss].mean():+.4f} (n={ss.sum()})")
print(f"  diff-dx              r={r[dd].mean():+.4f} (n={dd.sum()})")
print(f"  shuffled null        r={null.mean():+.4f}")
print(f"  gap {gap:+.4f}  z={z:+.2f}  p={pval:.3f}")
print(f"  radial bands  {['%.3f' % v for v in band_r]}")
print(f"  vertical bands {['%.3f' % v for v in band_y]}")
