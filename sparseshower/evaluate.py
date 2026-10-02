"""Evaluation of generated samples against truth.  NumPy only (scipy optional).

Run with `evaluate --runs NAME ... --name EVAL` (jobs/evaluate.sub) after
sampling, and `ae-eval` (jobs/ae_eval.sub) after AE training. Output goes to
<case>/eval/<name>/: summary.json, per_event.jsonl, passfail.md and (evaluate
only) summary.png.

Per event (`event_metrics`): active voxels, total photons, longitudinal (32
slabs) and radial (16 rings) photon profiles, top-5% photon share, per-voxel
logQ / nearest-neighbour / local-linearity distributions (kept as 512
quantiles), straight-track fraction (linearity > 0.95), box dimensions, and
the halo connectivity (definition below).

Halo connectivity:  voxels whose centre has hypot(x, y) > eval.halo_r
(0.5 m), 26-connected.  Reported: number of components, photon fraction of
the halo carried by components with >= eval.halo_min_voxels (20) voxels,
90th percentile of the component extent (bounding-box diagonal of the voxel
centres + one voxel), number of components with extent >= eval.halo_long_m
(0.6 m), and the halo's share of all photons.

Rungs (`evaluate`; a "rung" is one set of samples being evaluated):
    R0         truth half A vs truth half B (test split) - the baseline
    paired     sa_truth / sa_recon / sdedit / V1: each sample has an event id;
               per-event errors against that event, mean with bootstrap 95% CI,
               and the own/other ratio of the longitudinal L1
    all rungs  distribution level: W1 of every scalar statistic against truth
               half A, divided by the W1 of an equally large truth sample
               (half B) against A; C2ST AUC (logistic regression on
               standardized features + squares, k-fold, bootstrap CI);
               Frechet distance on standardized features
    full       memorization: nearest-neighbour distance of generated latents to
               train latents / the same for val latents (<< 1 = copying)

Metrics are always computed per event first and then aggregated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import digest, read_json, save_npz, write_json
from .data import load_event
from .geometry import OFFSETS_26, box_count, box_dimension, knn_within_grid, local_linearity, \
    neighbour_table, voxel_center
from .loader import case_dir, meta_for

NQ = 512
QGRID = (np.arange(NQ) + 0.5) / NQ
SCALARS = ("n_active", "q_total", "top5_share", "straight_frac", "lin_median", "nn_median",
           "box_d1", "box_d2", "box_d3", "box_d4", "halo_n_comp", "halo_big_frac", "halo_extent_p90",
           "halo_n_long", "halo_q_frac", "long_mean", "rad_mean", "logq_median", "logq_p90")
PAIRED_KEYS = ("n_rel_err", "q_rel_err", "long_l1", "rad_l1", "top5_rel_err", "straight_diff",
               "halo_long_diff", "halo_big_diff", "w1_logq", "w1_nn", "w1_lin")


# ----------------------------------------------------------- per event ---
def w1(a, b, n=NQ):
    """1-D Wasserstein-1 distance between two samples, from n matched quantiles."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    g = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, g) - np.quantile(b, g))))


def w1_q(qa, qb):
    """W1 between two distributions given as matched quantile vectors."""
    return float(np.mean(np.abs(np.asarray(qa) - np.asarray(qb))))


def profiles(ijk, q, c, z_slabs=32, r_rings=16):
    """Longitudinal (z slab) and radial (ring) photon profiles of one event, normalised."""
    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    centers = voxel_center(ijk, ranges, grid)
    frac = (ranges[2, 1] - centers[:, 2]) / (ranges[2, 1] - ranges[2, 0])
    zs = np.clip((frac * z_slabs).astype(int), 0, z_slabs - 1)
    r = np.hypot(centers[:, 0], centers[:, 1])
    rr = np.clip((np.sqrt(np.clip(r / float(c["macro"]["r_max"]), 0, 1)) * r_rings).astype(int), 0, r_rings - 1)
    q = np.asarray(q, dtype=np.float64)
    total = max(q.sum(), 1e-30)
    return (np.bincount(zs, weights=q, minlength=z_slabs) / total,
            np.bincount(rr, weights=q, minlength=r_rings) / total)


def connected_components(n, idx, mask):
    """Labels of the undirected graph given by a neighbour table."""
    rows = np.repeat(np.arange(n), idx.shape[1])[mask.reshape(-1) > 0]
    cols = idx.reshape(-1)[mask.reshape(-1) > 0]
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as cc

        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        return cc(g, directed=False)[1]
    except ImportError:
        labels = np.arange(n)
        while True:
            m = labels.copy()
            np.minimum.at(m, rows, labels[cols])
            np.minimum.at(m, cols, labels[rows])
            m = m[m]
            if np.array_equal(m, labels):
                return np.unique(labels, return_inverse=True)[1]
            labels = m


def halo_metrics(ijk, q, c):
    """Halo connectivity numbers of one event (components, photon share of long components)."""
    e = c["eval"]
    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = float(((ranges[:, 1] - ranges[:, 0]) / grid).mean())
    centres = voxel_center(ijk, ranges, grid)
    sel = np.hypot(centres[:, 0], centres[:, 1]) > float(e["halo_r"])
    q = np.asarray(q, dtype=np.float64)
    out = dict(halo_n_comp=0.0, halo_big_frac=0.0, halo_extent_p90=0.0, halo_n_long=0.0,
               halo_q_frac=float(q[sel].sum() / max(q.sum(), 1e-30)))
    if sel.sum() == 0:
        return out
    sub, pts, qs = np.asarray(ijk, dtype=np.int64)[sel], centres[sel], q[sel]
    idx, mask = neighbour_table(sub, grid, OFFSETS_26)
    lab = connected_components(len(sub), idx, mask)
    n_comp = int(lab.max()) + 1
    size = np.bincount(lab, minlength=n_comp)
    qc = np.bincount(lab, weights=qs, minlength=n_comp)
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(lab[order])) + 1]
    lo = np.minimum.reduceat(pts[order], starts, axis=0)
    hi = np.maximum.reduceat(pts[order], starts, axis=0)
    extent = np.linalg.norm(hi - lo, axis=1) + voxel
    out.update(halo_n_comp=float(n_comp),
               halo_big_frac=float(qc[size >= int(e["halo_min_voxels"])].sum() / max(qs.sum(), 1e-30)),
               halo_extent_p90=float(np.quantile(extent, 0.9)),
               halo_n_long=float((extent >= float(e["halo_long_m"])).sum()))
    return out


def event_metrics(ijk, q, off, c, k=8):
    """All per-event statistics of one voxel set (scalars, profiles, quantile vectors)."""
    grid = int(c["data"]["grid"])
    ranges = np.asarray(c["data"]["ranges"], dtype=np.float64)
    voxel = (ranges[:, 1] - ranges[:, 0]) / grid
    ijk = np.asarray(ijk, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    points = voxel_center(ijk, ranges, grid) + np.asarray(off, dtype=np.float64) * voxel
    factors = [f for f in (1, 2, 4, 8, 16) if grid % f == 0]
    bd = box_dimension(box_count(ijk, factors), factors) + [float("nan")] * 4
    dist, _ = knn_within_grid(points, ijk, grid, 2)
    nn = dist[:, 1]
    nn = nn[np.isfinite(nn)]
    lin = local_linearity(points, ijk, grid, k=k)
    logq = np.log(q + float(c["data"]["q_eps"]))
    order = np.sort(q)[::-1]
    share = np.cumsum(order) / max(order.sum(), 1e-30)
    long_p, rad_p = profiles(ijk, q, c)
    qv = lambda a: np.quantile(a, QGRID) if len(a) else np.full(NQ, np.nan)  # noqa: E731
    m = dict(n_active=float(len(ijk)), q_total=float(q.sum()),
             top5_share=float(share[min(len(share) - 1, int(0.05 * len(order)))]),
             straight_frac=float((lin > 0.95).mean()) if len(lin) else float("nan"),
             lin_median=float(np.median(lin)) if len(lin) else float("nan"),
             nn_median=float(np.median(nn)) if len(nn) else float("nan"),
             box_d1=bd[0], box_d2=bd[1], box_d3=bd[2], box_d4=bd[3],
             long_mean=float((np.arange(len(long_p)) + 0.5) @ long_p),
             rad_mean=float((np.arange(len(rad_p)) + 0.5) @ rad_p),
             logq_median=float(np.median(logq)), logq_p90=float(np.quantile(logq, 0.9)))
    m.update(halo_metrics(ijk, q, c))
    return dict(scalars=m, longitudinal=long_p, radial=rad_p, q_logq=qv(logq), q_nn=qv(nn), q_lin=qv(lin))


def features(m):
    """Event-level feature vector for C2ST / Frechet distance."""
    s = dict(m["scalars"])
    s["n_active"] = np.log10(max(s["n_active"], 1.0))
    s["q_total"] = np.log10(max(s["q_total"], 1e-3))
    return np.concatenate([[s[k] for k in SCALARS], m["longitudinal"], m["radial"]]).astype(np.float64)


# ------------------------------------------------------------ caching ---
def _pack(m):
    """Metrics dict -> arrays that can be saved in an .npz cache."""
    return dict(scalars=json.dumps(m["scalars"]), longitudinal=m["longitudinal"], radial=m["radial"],
                q_logq=m["q_logq"], q_nn=m["q_nn"], q_lin=m["q_lin"])


def _unpack(d):
    """Inverse of _pack."""
    return dict(scalars=json.loads(str(d["scalars"])), longitudinal=d["longitudinal"],
                radial=d["radial"], q_logq=d["q_logq"], q_nn=d["q_nn"], q_lin=d["q_lin"])


def metrics_cache_dir(c):
    """Cache directory of truth metrics, keyed by the settings that change them."""
    key = digest(dict(data=c["data"], macro=c["macro"], eval={k: c["eval"][k] for k in
                                                              ("halo_r", "halo_min_voxels", "halo_long_m")},
                      version=2))[:12]
    return Path(c["paths"]["output"]) / "metrics_cache" / key


def truth_metrics(c, eid):
    """Metrics of one truth test event (computed once, then read from the cache)."""
    path = metrics_cache_dir(c) / f"truth_{int(eid):05d}.npz"
    if path.exists():
        with np.load(path) as d:
            return _unpack({k: d[k] for k in d.files})
    ev = load_event(c["paths"]["processed"], eid)
    m = event_metrics(ev["ijk"], ev["q"], ev["off"], c)
    try:
        save_npz(path, **_pack(m))
    except OSError:
        pass
    return m


# ------------------------------------------------------------- statistics ---
def bootstrap_mean(x, n_boot, rng):
    """Mean of x with a bootstrap 95% confidence interval."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    means = x[rng.integers(len(x), size=(int(n_boot), len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(means, 0.025)),
                hi=float(np.quantile(means, 0.975)), n=int(len(x)))


def paired_errors(g, t):
    """Errors of one generated sample against its own truth event (relative / L1)."""
    gs, ts = g["scalars"], t["scalars"]
    rel = lambda a, b: float(abs(a / b - 1.0)) if b else float("nan")  # noqa: E731
    return dict(
        n_rel_err=rel(gs["n_active"], ts["n_active"]), q_rel_err=rel(gs["q_total"], ts["q_total"]),
        long_l1=float(np.abs(g["longitudinal"] - t["longitudinal"]).sum()),
        rad_l1=float(np.abs(g["radial"] - t["radial"]).sum()),
        top5_rel_err=rel(gs["top5_share"], ts["top5_share"]),
        straight_diff=gs["straight_frac"] - ts["straight_frac"],
        halo_long_diff=gs["halo_n_long"] - ts["halo_n_long"],
        halo_big_diff=gs["halo_big_frac"] - ts["halo_big_frac"],
        w1_logq=w1_q(g["q_logq"], t["q_logq"]), w1_nn=w1_q(g["q_nn"], t["q_nn"]),
        w1_lin=w1_q(g["q_lin"], t["q_lin"]))


def own_other(gen, truth_by_eid, rng, max_other=20):
    """Profile distance of each sample to its own truth vs to other truth events."""
    own, other = [], []
    eids = list(truth_by_eid)
    for g in gen:
        t = truth_by_eid[g["eid"]]
        own.append(np.abs(g["m"]["longitudinal"] - t["longitudinal"]).sum())
        others = [e for e in eids if e != g["eid"]]
        if others:
            pick = rng.choice(others, min(max_other, len(others)), replace=False)
            other.append(np.mean([np.abs(g["m"]["longitudinal"] - truth_by_eid[e]["longitudinal"]).sum()
                                  for e in pick]))
    return float(np.mean(own) / max(np.mean(other), 1e-30)) if other else float("nan")


def auc_score(y, s):
    """ROC AUC from labels and scores (rank formula, ties averaged)."""
    y = np.asarray(y, dtype=bool)
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logistic_fit(X, y, ridge=1.0, iters=30):
    """Ridge logistic regression by Newton iterations (used by c2st)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = ridge * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) + reg @ w
        h = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(h, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def c2st(Fa, Fb, folds, rng, n_boot=500):
    """Classifier two-sample test: AUC of logistic regression (features + squares), k-fold."""
    n = min(len(Fa), len(Fb))
    if n < 2 * folds:
        return dict(auc=float("nan"), lo=float("nan"), hi=float("nan"), n=int(n))
    A = Fa[rng.choice(len(Fa), n, replace=False)]
    B = Fb[rng.choice(len(Fb), n, replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(n), np.ones(n)]
    ok = np.all(np.isfinite(X), axis=0)
    X = X[:, ok]
    perm = rng.permutation(len(X))
    fold = np.empty(len(X), dtype=np.int64)
    fold[perm] = np.arange(len(X)) % int(folds)
    score = np.zeros(len(X))
    for f in range(int(folds)):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        Z = np.hstack([Z, Z ** 2 - 1.0])
        w = _logistic_fit(Z[tr], y[tr])
        score[te] = np.hstack([Z[te], np.ones((te.sum(), 1))]) @ w
    auc = auc_score(y, score)
    boots = []
    for _ in range(int(n_boot)):
        i = rng.integers(len(X), size=len(X))
        boots.append(auc_score(y[i], score[i]))
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    return dict(auc=auc, lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)), n=int(n))


def frechet(Fa, Fb, ref):
    """Frechet distance between two feature sets, standardised by a reference set."""
    if min(len(Fa), len(Fb), len(ref)) < 3:
        return float("nan")
    ok = np.all(np.isfinite(ref), axis=0) & np.all(np.isfinite(Fa), axis=0) & np.all(np.isfinite(Fb), axis=0)
    mu, sd = ref[:, ok].mean(0), ref[:, ok].std(0) + 1e-9
    A, B = (Fa[:, ok] - mu) / sd, (Fb[:, ok] - mu) / sd
    m1, m2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    ev, vec = np.linalg.eigh(s1)
    r1 = (vec * np.sqrt(np.maximum(ev, 0))) @ vec.T
    try:
        mid = np.linalg.eigvalsh(r1 @ s2 @ r1)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.maximum(mid, 0)).sum())


def distribution(gen_ms, A, B, rng, c):
    """W1 ratios per scalar, C2ST, Frechet.  A, B are truth metric lists (halves)."""
    out = {}
    for k in SCALARS:
        g = np.asarray([m["scalars"][k] for m in gen_ms], dtype=np.float64)
        a = np.asarray([m["scalars"][k] for m in A], dtype=np.float64)
        b = np.asarray([m["scalars"][k] for m in B], dtype=np.float64)
        g, a, b = g[np.isfinite(g)], a[np.isfinite(a)], b[np.isfinite(b)]
        wg = w1(g, a)
        n = min(len(g), len(b))
        ref = np.mean([w1(b[rng.choice(len(b), n, replace=False)], a) for _ in range(5)]) if n else float("nan")
        out[k] = dict(w1=wg, w1_r0=float(ref), ratio=float(wg / ref) if ref and ref > 0 else float("nan"),
                      mean=float(g.mean()) if len(g) else float("nan"),
                      truth_mean=float(a.mean()) if len(a) else float("nan"))
    Fg = np.stack([features(m) for m in gen_ms])
    Fa = np.stack([features(m) for m in A])
    res = dict(scalars=out, c2st=c2st(Fg, Fa, int(c["eval"]["c2st_folds"]), rng,
                                      n_boot=min(500, int(c["eval"]["bootstrap"]))),
               frechet=frechet(Fg, Fa, Fa), n=int(len(gen_ms)))
    res["profiles"] = dict(long_mean=np.mean([m["longitudinal"] for m in gen_ms], 0).tolist(),
                           long_sd=np.std([m["longitudinal"] for m in gen_ms], 0).tolist(),
                           rad_mean=np.mean([m["radial"] for m in gen_ms], 0).tolist(),
                           rad_sd=np.std([m["radial"] for m in gen_ms], 0).tolist())
    return res


# ---------------------------------------------------------------- rungs ---
def rung_samples(c, case, spec):
    """spec: a run name under <case>/samples, or 'v1:<dir>' for a v1 sample directory."""
    from .sample import load_run, load_v1_run

    if spec.startswith("v1:"):
        rows = load_v1_run(spec[3:])
        return dict(name="V1", chain="v1", rows=rows, settings=dict(source=spec[3:]))
    run_dir = case_dir(c, case) / "samples" / spec
    if not run_dir.exists():
        raise FileNotFoundError(f"no sample run {run_dir}")
    return dict(name=spec, chain=read_json(run_dir / "run.json")["chain"], rows=load_run(run_dir),
                settings=read_json(run_dir / "run.json"))


def memorization(c, case, rows, rng):
    """Nearest train-latent distance of generated latents vs of val latents (copying check)."""
    from .dit import nn_distance
    from .field_ae import LatentStore

    zs = [r["z"] for r in rows if "z" in r]
    if not zs:
        return None
    store = LatentStore(case_dir(c, case))
    n_tr = len(store.ids("train"))
    pick = np.sort(rng.choice(n_tr, min(n_tr, int(c["eval"]["memorization_train"])), replace=False))
    train = store.train_mu(pick)
    val = np.stack([store.standardize(store.posterior(e, "val")[0]) for e in store.ids("val")[:len(zs)]])
    g = np.stack(zs).astype(np.float64)
    d_gen, d_val = nn_distance(g, train), nn_distance(val, train)
    return dict(gen_median=float(np.median(d_gen)), val_median=float(np.median(d_val)),
                ratio=float(np.median(d_gen) / max(np.median(d_val), 1e-30)),
                frac_gen_closer_than_val_p5=float((d_gen < np.quantile(d_val, 0.05)).mean()),
                n_gen=int(len(g)), n_train=int(len(train)))


def evaluate(c, case, runs, name=None, truth_n=None, verbose=True):
    """Evaluate sample runs against the truth test events.

    Writes summary.json, per_event.jsonl, passfail.md and summary.png.
    """
    rng = np.random.default_rng(int(c["eval"]["seed"]))
    name = name or "eval_" + hashlib.sha1("|".join(runs).encode()).hexdigest()[:8]
    out_dir = case_dir(c, case) / "eval" / name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists; pick a new --name (evaluations are never overwritten)")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_for(c)
    test = [int(e) for e in (meta["split"]["test"] or meta["split"]["val"])]
    if truth_n:
        test = test[: int(truth_n)]
    say = print if verbose else (lambda *a, **k: None)
    say(f"truth metrics for {len(test)} test events (cached under {metrics_cache_dir(c)})")
    truth = {}
    for i, e in enumerate(test):
        truth[e] = truth_metrics(c, e)
        if verbose and (i + 1) % 100 == 0:
            say(f"  {i + 1}/{len(test)}")
    order = rng.permutation(len(test))
    A = [truth[test[i]] for i in order[: len(test) // 2]]
    B = [truth[test[i]] for i in order[len(test) // 2:]]
    summary = dict(case=case, name=name, truth_events=len(test), rungs={})
    summary["R0"] = distribution(B, A, B, rng, c)
    per_event = []
    for spec in runs:
        rs = rung_samples(c, case, spec)
        rows = rs["rows"]
        say(f"rung {rs['name']} ({rs['chain']}): {len(rows)} samples")
        for r in rows:
            r["m"] = event_metrics(r["ijk"], r["q"], r["off"], c)
        entry = dict(chain=rs["chain"], settings=rs["settings"], n=len(rows))
        paired = [r for r in rows if r["eid"] >= 0]
        if paired:
            for r in paired:
                if r["eid"] not in truth:
                    truth[r["eid"]] = truth_metrics(c, r["eid"])
            errs = [dict(eid=r["eid"], **paired_errors(r["m"], truth[r["eid"]])) for r in paired]
            per_event += [dict(rung=rs["name"], **e) for e in errs]
            entry["paired"] = {k: bootstrap_mean([e[k] for e in errs], int(c["eval"]["bootstrap"]), rng)
                               for k in PAIRED_KEYS}
            entry["own_other_long"] = own_other(paired, {r["eid"]: truth[r["eid"]] for r in paired}, rng)
        entry["distribution"] = distribution([r["m"] for r in rows], A, B, rng, c)
        if rs["chain"] == "full":
            entry["memorization"] = memorization(c, case, rows, rng)
        summary["rungs"][rs["name"]] = entry
    summary["passfail"] = passfail(summary, c)
    write_json(out_dir / "summary.json", _clean(summary))
    with open(out_dir / "per_event.jsonl", "w") as f:
        for row in per_event:
            f.write(json.dumps(_clean(row)) + "\n")
    (out_dir / "passfail.md").write_text(passfail_md(summary))
    summary["plot"] = plot(summary, out_dir / "summary.png")
    say(passfail_md(summary))
    return summary


def _clean(obj):
    """Make a nested result JSON-safe (NumPy types -> Python, NaN / inf -> None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# -------------------------------------------------------------- pass/fail ---
def passfail(summary, c):
    """Pass / fail rows for the metrics that have a threshold in configs/v2.yaml (`thresholds`)."""
    th = c.get("thresholds", {})
    rows = []
    dist_th = th.get("distribution", {})
    for name, r in summary["rungs"].items():
        d = r["distribution"]
        for k, v in d["scalars"].items():
            rows.append(dict(rung=name, metric=f"W1 ratio {k}", value=v["ratio"],
                             threshold=f"<= {dist_th.get('w1_ratio', 2.0)}",
                             ok=bool(v["ratio"] is not None and np.isfinite(v["ratio"])
                                     and v["ratio"] <= float(dist_th.get("w1_ratio", 2.0)))))
        auc = d["c2st"]["auc"]
        rows.append(dict(rung=name, metric="C2ST AUC", value=auc,
                         threshold=f"<= {dist_th.get('c2st_auc', 0.65)}",
                         ok=bool(np.isfinite(auc) and auc <= float(dist_th.get("c2st_auc", 0.65)))))
        pt = th.get("paired", {})
        if "paired" in r and r["chain"] in ("sa_truth", "sa_recon"):
            q = r["paired"]["q_rel_err"]["mean"]
            rows.append(dict(rung=name, metric="paired |Q ratio - 1|", value=q,
                             threshold=f"< {pt.get('q_total_rel_err', 0.02)}",
                             ok=bool(np.isfinite(q) and q < float(pt.get("q_total_rel_err", 0.02)))))
            if r["chain"] == "sa_recon":
                l1 = r["paired"]["long_l1"]["mean"]
                rows.append(dict(rung=name, metric="paired longitudinal L1", value=l1,
                                 threshold=f"<= {pt.get('long_l1', 0.03)}",
                                 ok=bool(np.isfinite(l1) and l1 <= float(pt.get("long_l1", 0.03)))))
        if r.get("memorization"):
            mr = r["memorization"]["ratio"]
            rows.append(dict(rung=name, metric="memorization ratio", value=mr,
                             threshold=f">= {th.get('memorization_ratio', 0.9)}",
                             ok=bool(mr >= float(th.get("memorization_ratio", 0.9)))))
    return rows


def _fmt(v):
    """Format a number for the markdown table."""
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def passfail_md(summary):
    """Markdown report of an evaluation (paired, distribution and pass / fail tables)."""
    lines = [f"# Evaluation {summary['name']} (case {summary['case']})", "",
             f"Truth: {summary['truth_events']} test events; R0 = half A vs half B.",
             "Thresholds are the proposed acceptance lines in configs/v2.yaml (`thresholds`).", ""]
    for name, r in summary["rungs"].items():
        lines += [f"## {name}  ({r['chain']}, {r['n']} samples)", ""]
        if "paired" in r:
            lines += ["Paired (mean, bootstrap 95% CI):", "", "| metric | mean | 95% CI |", "|---|---|---|"]
            for k, v in r["paired"].items():
                lines.append(f"| {k} | {_fmt(v['mean'])} | [{_fmt(v['lo'])}, {_fmt(v['hi'])}] |")
            lines += ["", f"own/other longitudinal L1: {_fmt(r.get('own_other_long'))} (1 = no event information)", ""]
        d = r["distribution"]
        lines += ["Distribution (W1 ratio = W1(gen, truth A) / W1(truth B, truth A), equal sizes):", "",
                  "| statistic | gen mean | truth mean | W1 ratio |", "|---|---|---|---|"]
        for k, v in d["scalars"].items():
            lines.append(f"| {k} | {_fmt(v['mean'])} | {_fmt(v['truth_mean'])} | {_fmt(v['ratio'])} |")
        cc = d["c2st"]
        lines += ["", f"C2ST AUC {_fmt(cc['auc'])} [{_fmt(cc['lo'])}, {_fmt(cc['hi'])}] (n = {cc['n']} per class); "
                      f"Frechet {_fmt(d['frechet'])} (R0 {_fmt(summary['R0']['frechet'])})", ""]
        if r.get("memorization"):
            m = r["memorization"]
            lines += [f"Memorization: median NN distance gen->train {_fmt(m['gen_median'])}, "
                      f"val->train {_fmt(m['val_median'])}, ratio {_fmt(m['ratio'])}", ""]
    lines += ["## Pass / fail", "", "| rung | metric | value | threshold | ok |", "|---|---|---|---|---|"]
    for row in summary["passfail"]:
        lines.append(f"| {row['rung']} | {row['metric']} | {_fmt(row['value'])} | {row['threshold']} | "
                     f"{'yes' if row['ok'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def plot(summary, path):
    """Summary figure of an evaluation (skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rungs = summary["rungs"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    r0 = summary["R0"]["profiles"]
    x = np.arange(len(r0["long_mean"]))
    ax[0].fill_between(x, np.subtract(r0["long_mean"], r0["long_sd"]), np.add(r0["long_mean"], r0["long_sd"]),
                       color="0.8", label="truth (half B) ±1σ")
    for name, r in rungs.items():
        ax[0].plot(x, r["distribution"]["profiles"]["long_mean"], label=name)
    ax[0].set_title("longitudinal photon share (32 slabs)")
    ax[0].legend(fontsize=7)
    xr = np.arange(len(r0["rad_mean"]))
    ax[1].fill_between(xr, np.subtract(r0["rad_mean"], r0["rad_sd"]), np.add(r0["rad_mean"], r0["rad_sd"]),
                       color="0.8")
    for name, r in rungs.items():
        ax[1].plot(xr, r["distribution"]["profiles"]["rad_mean"], label=name)
    ax[1].set_yscale("log")
    ax[1].set_title("radial photon share (16 rings)")
    names = list(rungs)
    auc = [rungs[n]["distribution"]["c2st"]["auc"] or np.nan for n in names]
    ax[2].bar(range(len(names)), auc)
    ax[2].axhline(0.5, color="k", lw=0.8)
    ax[2].set_xticks(range(len(names)), names, rotation=30, fontsize=7)
    ax[2].set_title("C2ST AUC (0.5 = indistinguishable)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------- AE eval ---
def evaluate_ae(c, case, n_events=128, checkpoint="best", name=None, allow_cpu=False):
    """Acceptance metrics of the FieldAE on val events."""
    import torch

    from .common import device_for
    from .field_ae import ae_event_metrics, decoded_field, event_field, load_ae
    from .fields import cell_profile_index, load_field_stats

    device = device_for(allow_cpu)
    out_dir = case_dir(c, case) / "eval" / (name or f"ae_{checkpoint}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} exists; pick a new --name")
    out_dir.mkdir(parents=True, exist_ok=True)
    ae, state = load_ae(case_dir(c, case), checkpoint, device, c)
    stats = load_field_stats(c)
    pidx = cell_profile_index(c, int(c["field"]["base_grid"]))
    ids = [int(e) for e in meta_for(c)["split"]["val"]][: int(n_events)]
    rows = []
    with torch.no_grad():
        for e in ids:
            f, sums, counts = event_field(load_event(c["paths"]["processed"], e), c, stats)
            out, _, _ = ae(torch.as_tensor(f[None], device=device), sample=False)
            rows.append(dict(eid=e, **ae_event_metrics(decoded_field(out)[0].cpu().numpy(), counts, sums,
                                                        c, stats, pidx)))
    rng = np.random.default_rng(int(c["eval"]["seed"]))
    th = c.get("thresholds", {}).get("ae", {})
    res = dict(case=case, checkpoint=checkpoint, ae_step=int(state["step"]), n_events=len(rows), metrics={},
               passfail=[])
    for k in ("lost_photon_frac", "n48_rel_err", "field_long_l1", "field_rad_l1", "q_total_rel_err",
              "extra_cells_frac"):
        res["metrics"][k] = bootstrap_mean([r[k] for r in rows], int(c["eval"]["bootstrap"]), rng)
        if k in th:
            res["passfail"].append(dict(metric=k, value=res["metrics"][k]["mean"], threshold=f"< {th[k]}",
                                        ok=bool(res["metrics"][k]["mean"] < float(th[k]))))
    write_json(out_dir / "summary.json", _clean(res))
    with open(out_dir / "per_event.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(_clean(r)) + "\n")
    md = [f"# AE acceptance (case {case}, ae_{checkpoint}.pt step {res['ae_step']}, {len(rows)} val events)", "",
          "| metric | mean | 95% CI | threshold | ok |", "|---|---|---|---|---|"]
    th_map = {p["metric"]: p for p in res["passfail"]}
    for k, v in res["metrics"].items():
        p = th_map.get(k)
        md.append(f"| {k} | {_fmt(v['mean'])} | [{_fmt(v['lo'])}, {_fmt(v['hi'])}] | "
                  f"{p['threshold'] if p else '-'} | {('yes' if p['ok'] else 'NO') if p else '-'} |")
    (out_dir / "passfail.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res
