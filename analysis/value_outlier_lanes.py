#!/usr/bin/env python3
"""Are value-sensitive layers just OUTLIER layers? Test FP16 outlier lanes for values.

The per-layer probe said value sensitivity is outlier-driven: the layers that hurt
at INT3 are the high-outlier layers ("one value ruins the scale" -- the per-token
ruler gets stretched by one hot channel and the other 127 round to mush).
mixed_top6 fixes this by giving ALL 128 channels an extra bit on 6 layers
(model-specific list, re-probe per model). This tests the cheaper structural fix:
give values the exact medicine ChannelQuant already gives keys -- pull the top-k
outlier channels per head into an FP16 lane, quantize the remaining channels at
INT3, and (the actual mechanism) EXCLUDE the lane channels from the per-token
scale so the ruler is computed over calm channels only.

Phase 0 (cheap, one forward pass over wikitext chunks): per-layer probe on THIS
model -- INT3 value error with/without lanes, INT3-per-channel key error, outlier
ratios. Auto-derives the sensitive-layer sets (top-6 by INT3 error) and doubles as
the missing 1.5B KEY probe ("keys flat across layers" was measured on 0.5B only,
before the key/value fragility asymmetry flipped at 1.5B).

Phase 1 (HellaSwag acc_norm): ten arms -- fp16 / uniform4 / uniform3 /
mixed_top6 (re-test at n=1000) / INT3+2 lanes all layers / INT3+4 lanes /
lanes only on sensitive layers / per-layer keys (top-6 CQ4+, rest CQ3+) /
WHT-rotated INT3 values / WHT-rotated INT2 values (stretch) / WHT + Lloyd-Max
INT3 (the full TurboQuant value path: post-rotation rows are ~gaussian, where
the Lloyd-Max codebook is MSE-optimal vs uniform ticks) / Lloyd-Max without
the rotation (ablation: codebook contribution vs spin contribution).
--arms selects: 'core' = the promised follow-up set (baselines, rotation,
lloyd-max, keys below 4 bit per layer), 'all', or a comma list.

The WHT arms complete the outlier-medicine triangle: lanes ISOLATE the hot
channels (FP16 lane), grouped-INT2 LOCALIZES them (graded_grouped2.py), the
Walsh-Hadamard rotation SPREADS them -- spin each value row before quantizing so
no single channel sets the per-token scale, un-spin on read (H is self-inverse).
This is the retired TurboQuant+ rotation applied to VALUES ONLY: the -0.10 GQA
collapse that killed TurboQuant+ was the rotation smearing the per-channel KEY
outlier structure; values are per-token (each row self-contained), so that
failure mode structurally cannot occur. Keys stay untouched ChannelQuant.
Rotation is compute, not storage: bit accounting is unchanged (lanes=0).

Same transformers-5.x AttentionInterface plumbing and gotchas as
graded_2bit.py / graded_grouped2.py: self-built causal mask, fp32 QK^T
(fp16 -> NaN at D=128), quantization after GQA repeat_interleave. Bit accounting
matches the prior studies (payload bits; per-token/per-channel scale overhead and
static lane indices not counted, same as CQ-4 counted as 4.0).

Colab-safe: writes the JSON after EVERY arm and mirrors it to Drive if mounted;
on restart it reloads (local first, then Drive) and skips finished arms.
"""
import argparse, json, math, os, shutil
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
G_TOK = 128          # token-group size for per-channel key scales (matches KVCE)
K_OUT_KEYS = 2       # FP16 outlier lanes keys already have (CQ-4+)

# per-layer plan, filled in per arm: lists indexed by layer_idx
CFG = {"fp16": True, "v_bits": None, "v_lanes": None, "v_rot": None, "v_rht": None,
       "v_lloyd": None, "k_bits": None, "probe": False}
PROBE = {}           # layer_idx -> accumulated stats
_SIGNS = {}          # D -> fixed random ±1 sign vector (the RHT's diagonal)

# Lloyd-Max codebook levels for N(0,1) (Max 1960), positive half; negatives mirror.
# Post-WHT value rows are ~gaussian, where this codebook is the MSE-optimal
# quantizer -- this is TurboQuant's value quantizer (norm -> WHT -> Lloyd-Max).
_LLOYD_LEVELS = {3: [0.2451, 0.7560, 1.3439, 2.1520], 2: [0.4528, 1.5104]}


def _fwht(x):
    """Orthonormal fast Walsh-Hadamard on the last dim (D must be a power of 2:
    64 on 0.5B, 128 on 1.5B). Self-inverse: _fwht(_fwht(x)) == x."""
    D = x.shape[-1]
    h = 1
    while h < D:
        y = x.view(*x.shape[:-1], -1, h * 2)
        a, b = y[..., :h], y[..., h:]
        x = torch.cat((a + b, a - b), dim=-1).view(*x.shape)
        h *= 2
    return x / math.sqrt(D)


def _signs(D, device):
    """Fixed random ±1 per channel (seeded, same vector everywhere -- in RTL this
    is one programmable 128-bit sign register in front of the butterfly). Breaks
    the fixed-H worst case: a flat row (big DC component) that H would CONCENTRATE
    into one hot coordinate instead of spreading."""
    if D not in _SIGNS:
        g = torch.Generator().manual_seed(1234)
        _SIGNS[D] = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    return _SIGNS[D].to(device)


def _q_lloyd(x, bits):
    """Per-token RMS scale + Lloyd-Max N(0,1) codebook lookup on the last dim."""
    pos = _LLOYD_LEVELS[bits]
    lv = torch.tensor([-l for l in reversed(pos)] + pos, device=x.device)
    th = (lv[1:] + lv[:-1]) / 2
    rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=EPS)
    return lv[torch.bucketize(x / rms, th)] * rms


def _q_values(v, bits, lanes, rot=False, lloyd=False, rht=False):
    """Per-token symmetric quant with optional FP16 outlier-channel lanes.
    Lane channels are picked per (B,H) by amax over tokens and are EXCLUDED from
    the per-token scale (they no longer stretch the ruler), then kept at fp16.
    rot spins the row with the WHT first (unspun after); rht flips the fixed
    random signs before the spin (randomized Hadamard: v -> (v*s)H, inverse
    (x H)*s since s^2=1 and H is self-inverse -- deferred unspin still works,
    the signs commute out the same way); lloyd swaps the uniform round for the
    gaussian-optimal Lloyd-Max codebook (RMS row scale)."""
    if bits >= 16:
        return v
    vf = v.float()
    B, H, T, D = vf.shape
    if rot:
        # SPREAD medicine: spin the row so the outlier stops setting the scale,
        # quantize the flat row, spin back (H orthonormal -> self-inverse).
        # Mutually exclusive with lanes (after rotation there are no hot channels).
        s = _signs(D, v.device) if rht else None
        x = _fwht(vf * s) if rht else _fwht(vf)
        inner = _q_lloyd(x, bits) if lloyd else _q_values(x, bits, 0)
        out = _fwht(inner)
        if rht:
            out = out * s
        return out.to(v.dtype)
    if lloyd:
        # ablation control: Lloyd-Max WITHOUT the rotation, to separate how much
        # of any win comes from the codebook vs from the spin
        return _q_lloyd(vf, bits).to(v.dtype)
    if lanes > 0:
        idx = vf.abs().amax(2).topk(lanes, -1).indices                 # [B,H,lanes]
        om = torch.zeros(B, H, D, dtype=torch.bool, device=v.device)
        om.scatter_(-1, idx, True)
        ome = om.unsqueeze(2)                                          # [B,H,1,D]
        base = vf.masked_fill(ome, 0.0)
    else:
        ome, base = None, vf
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    amax = base.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=EPS)
    q = torch.round(base / scale).clamp(qmin, qmax) * scale
    if ome is not None:
        q = torch.where(ome.expand_as(q), vf.to(torch.float16).float(), q)
    return q.to(v.dtype)


def _q_keys(k, bits):
    """CQ-style keys, bit-parameterized: per-channel scale within G_TOK-token
    groups + K_OUT_KEYS FP16 outlier channels (amax over T). bits=4 == CQ-4+."""
    if bits >= 16:
        return k
    B, H, T, D = k.shape
    kf = k.float(); out = torch.empty_like(kf)
    out_idx = kf.abs().amax(2).topk(K_OUT_KEYS, -1).indices
    om = torch.zeros(B, H, D, dtype=torch.bool, device=k.device)
    om.scatter_(-1, out_idx, True)
    qmax = (1 << (bits - 1)) - 1; qmin = -(1 << (bits - 1))
    for a in range(0, T, G_TOK):
        b = min(a + G_TOK, T); grp = kf[:, :, a:b, :]
        s = torch.clamp(grp.abs().amax(2, keepdim=True) / qmax, min=EPS)
        out[:, :, a:b, :] = torch.round(grp / s).clamp(qmin, qmax) * s
    out = torch.where(om.unsqueeze(2).expand(B, H, T, D), k.to(torch.float16).float(), out)
    return out.to(k.dtype)


def _relerr(x, q):
    return (x - q).norm().item() / max(x.norm().item(), 1e-12)


def _probe_layer(li, k, v):
    kf, vf = k.float(), v.float()
    st = PROBE.setdefault(li, {"n": 0, "v3": 0.0, "v3_l2": 0.0, "v4": 0.0,
                               "k3": 0.0, "k4": 0.0, "v_outlier": 0.0, "k_outlier": 0.0})
    st["n"] += 1
    st["v3"] += _relerr(vf, _q_values(v, 3, 0).float())
    st["v3_l2"] += _relerr(vf, _q_values(v, 3, 2).float())
    st["v4"] += _relerr(vf, _q_values(v, 4, 0).float())
    st["k3"] += _relerr(kf, _q_keys(k, 3).float())
    st["k4"] += _relerr(kf, _q_keys(k, 4).float())
    for x, key in ((vf, "v_outlier"), (kf, "k_outlier")):
        am = x.abs().amax(2)                                           # [B,H,D]
        st[key] += (am.amax(-1) / am.median(-1).values.clamp(min=1e-12)).mean().item()


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    li = getattr(module, "layer_idx", 0)
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], key.shape[-2]
    if CFG["probe"]:
        _probe_layer(li, key, value)
    elif not CFG["fp16"]:
        key = _q_keys(key, CFG["k_bits"][li])
        value = _q_values(value, CFG["v_bits"][li], CFG["v_lanes"][li],
                          rot=CFG["v_rot"][li] if CFG["v_rot"] else False,
                          lloyd=CFG["v_lloyd"][li] if CFG["v_lloyd"] else False,
                          rht=CFG["v_rht"][li] if CFG["v_rht"] else False)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    scores = scores.masked_fill(~(j <= i), float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(A.to(query.dtype), value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("ollanes", attn)


def run_probe(model, tok, chunks=8, ctx=512):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    CFG["probe"] = True; PROBE.clear()
    with torch.no_grad():
        for c in range(chunks):
            seg = ids[c * ctx:(c + 1) * ctx].unsqueeze(0).cuda()
            model(seg)
    CFG["probe"] = False
    out = {}
    for li, st in sorted(PROBE.items()):
        n = st["n"]
        out[li] = {k: round(v / n, 5) for k, v in st.items() if k != "n"}
    return out


def sensitive_set(probe, metric, top=6):
    order = sorted(probe, key=lambda li: probe[li][metric], reverse=True)
    return sorted(order[:top])


def build_plan(name, L, sens_v, sens_k):
    v_bits = [4] * L; v_lanes = [0] * L; v_rot = [False] * L; v_rht = [False] * L
    v_lloyd = [False] * L; k_bits = [4] * L; fp16 = False
    if name == "fp16":
        fp16 = True
    elif name == "uniform_v4":
        pass
    elif name == "uniform_v3":
        v_bits = [3] * L
    elif name == "mixed_top6":
        v_bits = [4 if l in sens_v else 3 for l in range(L)]
    elif name == "ol_all_k2":
        v_bits = [3] * L; v_lanes = [2] * L
    elif name == "ol_all_k4":
        v_bits = [3] * L; v_lanes = [4] * L
    elif name == "ol_top6_k2":
        v_bits = [3] * L; v_lanes = [2 if l in sens_v else 0 for l in range(L)]
    elif name == "kmix_top6_v4":
        k_bits = [4 if l in sens_k else 3 for l in range(L)]
    elif name == "wht_all_v3":
        v_bits = [3] * L; v_rot = [True] * L
    elif name == "wht_all_v2":
        v_bits = [2] * L; v_rot = [True] * L
    elif name == "wht_lloyd_v3":     # full TurboQuant value path: WHT + Lloyd-Max
        v_bits = [3] * L; v_rot = [True] * L; v_lloyd = [True] * L
    elif name == "lloyd_v3":         # codebook without the spin (ablation control)
        v_bits = [3] * L; v_lloyd = [True] * L
    elif name == "rht_all_v3":       # randomized Hadamard (Chaithu's RTL question)
        v_bits = [3] * L; v_rot = [True] * L; v_rht = [True] * L
    else:
        raise ValueError(name)
    return {"fp16": fp16, "v_bits": v_bits, "v_lanes": v_lanes, "v_rot": v_rot,
            "v_rht": v_rht, "v_lloyd": v_lloyd, "k_bits": k_bits}


def avg_bits(plan, D=128):
    if plan["fp16"]:
        return 16.0, 16.0
    L = len(plan["v_bits"])
    vb = sum(((D - la) * b + la * 16.0) / D
             for b, la in zip(plan["v_bits"], plan["v_lanes"])) / L
    kb = sum(plan["k_bits"]) / L
    return round(vb, 4), round(kb, 4)


ARMS = ["fp16", "uniform_v4", "uniform_v3", "mixed_top6",
        "ol_all_k2", "ol_all_k4", "ol_top6_k2", "kmix_top6_v4",
        "wht_all_v3", "wht_all_v2", "wht_lloyd_v3", "lloyd_v3", "rht_all_v3"]

# the promised follow-up set: baselines + rotation + fixed-vs-randomized +
# lloyd-max + keys-below-4-bit. ordered so a Colab timeout still leaves the
# load-bearing rows finished.
CORE = ["fp16", "uniform_v4", "uniform_v3", "wht_all_v3", "rht_all_v3",
        "wht_lloyd_v3", "lloyd_v3", "kmix_top6_v4"]


def save(R, path, drive_dir):
    with open(path, "w") as f:
        json.dump(R, f, indent=2)
    if drive_dir and os.path.isdir("/content/drive"):
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(path, os.path.join(drive_dir, os.path.basename(path)))


def load_prior(path, drive_dir):
    for p in (path, os.path.join(drive_dir, os.path.basename(path)) if drive_dir else None):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="value_outlier_lanes_result.json")
    ap.add_argument("--drive_dir", default="/content/drive/MyDrive/tiu_runs")
    ap.add_argument("--arms", default="core",
                    help="'core' (promised follow-ups), 'all', or comma list of arm names")
    a = ap.parse_args()
    arms = CORE if a.arms == "core" else ARMS if a.arms == "all" else a.arms.split(",")
    for nm in arms:
        if nm not in ARMS:
            raise SystemExit(f"unknown arm: {nm}")
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 attn_implementation="ollanes").cuda().eval()
    L = model.config.num_hidden_layers
    D = model.config.hidden_size // model.config.num_attention_heads

    R = load_prior(a.out, a.drive_dir) or {"model": a.model, "n": a.n, "results": {}}
    if R.get("model") != a.model or R.get("n") != a.n:
        print(f"prior JSON is for {R.get('model')}/n={R.get('n')} -- starting fresh")
        R = {"model": a.model, "n": a.n, "results": {}}

    if "probe" not in R:
        print("=== phase 0: per-layer probe (wikitext, 8x512 tokens) ===")
        probe = run_probe(model, tok)
        R["probe"] = {str(li): st for li, st in probe.items()}
        save(R, a.out, a.drive_dir)
    probe = {int(li): st for li, st in R["probe"].items()}
    sens_v = sensitive_set(probe, "v3"); sens_k = sensitive_set(probe, "k3")
    R["sens_v"], R["sens_k"] = sens_v, sens_k
    print(f"{'layer':>5} {'v3err':>8} {'v3+2lane':>9} {'v4err':>8} {'k3err':>8} {'k4err':>8} {'v_outl':>7} {'k_outl':>7}")
    for li in sorted(probe):
        st = probe[li]
        tag = (" V" if li in sens_v else "") + (" K" if li in sens_k else "")
        print(f"{li:>5} {st['v3']:>8.4f} {st['v3_l2']:>9.4f} {st['v4']:>8.4f}"
              f" {st['k3']:>8.4f} {st['k4']:>8.4f} {st['v_outlier']:>7.1f} {st['k_outlier']:>7.1f}{tag}")
    print(f"sensitive value layers (top-6 by v3 err): {sens_v}")
    print(f"sensitive key   layers (top-6 by k3 err): {sens_k}")

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
    for name in arms:
        if name in R["results"]:
            print(f"  {name:14s} already done -- skipping"); continue
        plan = build_plan(name, L, set(sens_v), set(sens_k))
        CFG.update(plan)
        torch.manual_seed(0)
        out = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"], limit=a.n, bootstrap_iters=0)
        acc = out["results"]["hellaswag"]["acc_norm,none"]
        vb, kb = avg_bits(plan, D)
        mem = round((vb + kb) / 32.0, 4)
        R["results"][name] = {"acc_norm": acc, "avg_value_bits": vb, "avg_key_bits": kb, "mem": mem}
        save(R, a.out, a.drive_dir)
        print(f"  {name:14s} acc={acc:.4f}  v={vb:.3f}b k={kb:.3f}b mem={mem:.4f}")

    base = R["results"]["fp16"]["acc_norm"]
    print(f"\n=== value outlier lanes vs per-layer protection ({a.model}, n={a.n}) ===")
    for name in ARMS:
        r = R["results"].get(name)
        if not r:
            continue
        r["delta_vs_fp16"] = round(r["acc_norm"] - base, 4)
        print(f"  {name:14s} mem={r['mem']:.4f}  acc={r['acc_norm']:.4f}  Δ={r['delta_vs_fp16']:+.4f}")
    save(R, a.out, a.drive_dir)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
