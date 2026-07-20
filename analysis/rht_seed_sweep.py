#!/usr/bin/env python3
"""Design-time sign-vector ("seed") selection for the randomized Hadamard.

The RHT sign pattern ends up in a boot-time register (worst case: frozen into
silicon), so picking it is a one-shot design decision. This sweeps candidate
random ±1 sign vectors and, for each, verifies the rotated value rows do NOT
concentrate energy into any single coordinate on any layer/head -- concentration
is the exact hot-channel-hijacks-the-scale disease the rotation exists to cure,
reintroduced by an unlucky pattern. Run it across BOTH target models (Qwen2-1.5B
and Llama-3.2-1B) and freeze a seed that is spike-free on both.

Per (seed, layer, head), over wikitext forward passes (no benchmark evals --
this is cheap, ~10 min/model on a T4):

  * conc  = amax(|row|) / rms(row) after rotation, averaged over tokens.
            sqrt(D) = all energy in one coordinate (the disease);
            a gaussian row sits around ~3 for D=128. Lower = flatter = better.
  * rel3  = INT3 per-token quant relative error after rotation (accuracy proxy).

Candidates: seed index 0 = FIXED H (all +1 signs, the baseline Chaithu measured)
plus --n_seeds random vectors. Verdict per seed = worst (layer,head) conc and
worst rel3 across all layers/heads of all models; recommendation = the random
seed with the lowest worst-case conc (rel3 tiebreak). If fixed H's row beats
every random seed on rel3, that is the "fixed is ~1pt better" claim showing up
at the tensor level -- freeze all +1s and keep the register as insurance.

meta-llama/Llama-3.2-1B is gated: `huggingface-cli login` (or HF_TOKEN) first.
Both models' head_dim is a power of 2 (128 / 64), which the FWHT requires.

Colab-safe: JSON written after every model, mirrored to Drive if mounted.
"""
import argparse, gc, json, math, os, shutil
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

EPS = 2.0 ** -14
STATE = {"on": False}
SIGNVECS = []        # list of [D] ±1 tensors; index 0 = all +1 (fixed H)
STATS = {}           # (layer, seed_idx) -> {"conc_sum": [H], "rel_sum": [H], "n": int}


def _fwht(x):
    D = x.shape[-1]
    h = 1
    while h < D:
        y = x.view(*x.shape[:-1], -1, h * 2)
        a, b = y[..., :h], y[..., h:]
        x = torch.cat((a + b, a - b), dim=-1).view(*x.shape)
        h *= 2
    return x / math.sqrt(D)


def _measure(li, v):
    vf = v.float()                                         # [B,H,T,D] (KV heads)
    for si, s in enumerate(SIGNVECS):
        x = _fwht(vf * s.to(vf.device))
        am = x.abs().amax(-1)                              # [B,H,T]
        rms = x.pow(2).mean(-1).sqrt().clamp(min=1e-12)
        conc = (am / rms).mean((0, 2))                     # [H] mean over tokens
        scale = (am / 3).clamp(min=EPS).unsqueeze(-1)      # INT3: qmax=3
        qx = torch.round(x / scale).clamp(-4, 3) * scale
        rel = ((x - qx).norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-12)).mean((0, 2))
        st = STATS.setdefault((li, si), {"conc_sum": torch.zeros_like(conc.cpu()),
                                         "rel_sum": torch.zeros_like(rel.cpu()), "n": 0})
        st["conc_sum"] += conc.cpu(); st["rel_sum"] += rel.cpu(); st["n"] += 1


def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if STATE["on"]:
        _measure(getattr(module, "layer_idx", 0), value)
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, 1); value = value.repeat_interleave(n_rep, 1)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    Tq, Tk = query.shape[-2], key.shape[-2]
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    i = torch.arange(Tq, device=scores.device).unsqueeze(-1)
    j = torch.arange(Tk, device=scores.device).unsqueeze(0)
    scores = scores.masked_fill(~(j <= i), float("-inf"))
    A = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(A.to(query.dtype), value)
    return out.transpose(1, 2).contiguous(), A


AttentionInterface.register("rhtsweep", attn)


def make_signvecs(D, n_seeds):
    vecs = [torch.ones(D)]                                 # index 0 = fixed H
    for seed in range(1, n_seeds + 1):
        g = torch.Generator().manual_seed(seed)
        vecs.append((torch.randint(0, 2, (D,), generator=g) * 2 - 1).float())
    return vecs


def sweep_model(model_id, n_seeds, chunks, ctx):
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 attn_implementation="rhtsweep").cuda().eval()
    D = model.config.hidden_size // model.config.num_attention_heads
    SIGNVECS.clear(); SIGNVECS.extend(make_signvecs(D, n_seeds))
    STATS.clear()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    STATE["on"] = True
    with torch.no_grad():
        for c in range(chunks):
            model(ids[c * ctx:(c + 1) * ctx].unsqueeze(0).cuda())
    STATE["on"] = False

    out = {"head_dim": D, "gaussian_ref_conc": round(math.sqrt(2 * math.log(D)), 2),
           "seeds": {}}
    n_layers = 1 + max(li for li, _ in STATS)
    for si in range(len(SIGNVECS)):
        worst_conc, worst_rel, wl, wh = -1.0, -1.0, -1, -1
        conc_mean = rel_mean = 0.0
        for li in range(n_layers):
            st = STATS[(li, si)]
            conc = st["conc_sum"] / st["n"]; rel = st["rel_sum"] / st["n"]
            conc_mean += conc.mean().item() / n_layers
            rel_mean += rel.mean().item() / n_layers
            c, h = conc.max(0)
            if c.item() > worst_conc:
                worst_conc, wl, wh = c.item(), li, h.item()
            worst_rel = max(worst_rel, rel.max().item())
        name = "FIXED_H" if si == 0 else f"seed_{si}"
        out["seeds"][name] = {"worst_conc": round(worst_conc, 3),
                              "worst_conc_layer": wl, "worst_conc_head": wh,
                              "mean_conc": round(conc_mean, 3),
                              "worst_rel3": round(worst_rel, 4),
                              "mean_rel3": round(rel_mean, 4)}
    del model; gc.collect(); torch.cuda.empty_cache()
    return out


def save(R, path, drive_dir):
    with open(path, "w") as f:
        json.dump(R, f, indent=2)
    if drive_dir and os.path.isdir("/content/drive"):
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy(path, os.path.join(drive_dir, os.path.basename(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2-1.5B,meta-llama/Llama-3.2-1B")
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", default="rht_seed_sweep.json")
    ap.add_argument("--drive_dir", default="/content/drive/MyDrive/tiu_runs")
    a = ap.parse_args()

    R = {"models": {}}
    for mid in a.models.split(","):
        print(f"=== sweeping {mid} ===")
        R["models"][mid] = sweep_model(mid, a.n_seeds, a.chunks, a.ctx)
        save(R, a.out, a.drive_dir)

    # cross-model verdict: a seed is judged by its WORST (layer,head) on ANY model
    names = list(next(iter(R["models"].values()))["seeds"])
    table = {}
    for nm in names:
        table[nm] = {
            "worst_conc": max(m["seeds"][nm]["worst_conc"] for m in R["models"].values()),
            "worst_rel3": max(m["seeds"][nm]["worst_rel3"] for m in R["models"].values()),
            "mean_rel3": sum(m["seeds"][nm]["mean_rel3"] for m in R["models"].values())
                         / len(R["models"]),
        }
    rand = {nm: t for nm, t in table.items() if nm != "FIXED_H"}
    winner = min(rand, key=lambda nm: (round(rand[nm]["worst_conc"], 2),
                                       rand[nm]["mean_rel3"]))
    R["cross_model"] = table
    R["recommended_random_seed"] = winner
    R["fixed_h_beats_all_random_on_rel3"] = all(
        table["FIXED_H"]["mean_rel3"] <= rand[nm]["mean_rel3"] for nm in rand)
    save(R, a.out, a.drive_dir)

    print(f"\n{'pattern':>10} {'worst_conc':>11} {'worst_rel3':>11} {'mean_rel3':>10}")
    for nm in names:
        t = table[nm]
        mark = "  <-- winner (random)" if nm == winner else ""
        print(f"{nm:>10} {t['worst_conc']:>11.3f} {t['worst_rel3']:>11.4f}"
              f" {t['mean_rel3']:>10.4f}{mark}")
    print(f"\ngaussian reference conc (flat row target): "
          f"{ {m: r['gaussian_ref_conc'] for m, r in R['models'].items()} }")
    print(f"recommended random pattern: {winner} "
          f"(freeze only if it also holds up in the accuracy arm: "
          f"value_outlier_lanes.py --arms rht_all_v3 --rht_seed {winner.split('_')[-1]})")
    if R["fixed_h_beats_all_random_on_rel3"]:
        print("NOTE: FIXED_H has the best mean rel3 -- consistent with 'fixed is "
              "better' at the tensor level; if the accuracy arms agree, freeze all +1s.")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
