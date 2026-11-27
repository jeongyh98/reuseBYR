"""Build the Bridge demonstration libraries from successful rollouts collected with
`pi0_bridge_client.py --collect`.

  bridge_library.npz       emb (N,1536), chunk (N,4,7), task (N,)     -> selector score
  bridge_espr_library.npz  + act_pred, event_starts, episode/frame ids -> event-schema retrieval
  bridge_action_mlp.pt     embedding -> z-normalized flattened chunk (action key of retrieval)

Usage (lerobot env):
  python simpler_bridge/build_bridge_library.py --collected data/bridge_collect_{carrot,spoon,stack}.npz \
      --out_dir data
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from encoder import DinoSigLIPEncoder  # noqa: E402
from mlp import MLP  # noqa: E402

TASK_ID = {"carrot": 0, "spoon": 1, "stack": 2}


def bridge_events(ch, ep, fi, cos_thresh=0.3, mag_min=0.02):
    """Event starts for Bridge chunks, whose gripper entry is a command in [0,1] rather than a
    delta: episode start, gripper crossing 0.5, or a motion-direction change of the step-1 xyz."""
    grip, mot = ch[:, 1, 6], ch[:, 1, :3].astype(np.float64)
    ev = np.zeros(len(ch), bool)
    prev_e = prev_grip = prev_mot = None
    for idx in np.lexsort((fi, ep)):
        if ep[idx] != prev_e:
            ev[idx] = True
        elif (prev_grip > 0.5) != (grip[idx] > 0.5):
            ev[idx] = True
        else:
            m, pm = mot[idx], prev_mot
            nm, npm = np.linalg.norm(m), np.linalg.norm(pm)
            if nm > mag_min and npm > mag_min and (m @ pm) / (nm * npm) < cos_thresh:
                ev[idx] = True
        prev_e, prev_grip, prev_mot = ep[idx], grip[idx], mot[idx]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", nargs="+", required=True,
                    help="files named bridge_collect_<task>.npz from pi0_bridge_client.py --collect")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    torch.manual_seed(0)

    imgs, chunks, tasks, epis, fis, ep_off = [], [], [], [], [], 0
    for f in sorted(args.collected):
        task = Path(f).stem.split("_")[-1]
        d = np.load(f)
        for e in np.unique(d["episode_index"]):
            n = int((d["episode_index"] == e).sum())
            fis.append(np.arange(n, dtype=np.int64))
            epis.append(np.full(n, ep_off, np.int64))
            ep_off += 1
        imgs.append(d["img"]); chunks.append(d["chunk"])
        tasks.append(np.full(len(d["img"]), TASK_ID[task], np.int64))
        print(f"{f}: {len(d['img'])} frames ({task})")
    imgs, chunks = np.concatenate(imgs), np.concatenate(chunks).astype(np.float32)
    tasks, episode_index, frame_index = np.concatenate(tasks), np.concatenate(epis), np.concatenate(fis)
    N, H, da = chunks.shape

    enc = DinoSigLIPEncoder(device="cuda")
    with torch.no_grad():
        emb = np.concatenate([enc([imgs[j] for j in range(i, min(i + 64, N))]).float().cpu().numpy()
                              for i in range(0, N, 64)]).astype(np.float32)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(f"{args.out_dir}/bridge_library.npz", emb=emb, chunk=chunks, task=tasks)

    # action MLP: L2-normalized embedding -> z-normalized flattened chunk
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    Y = chunks.reshape(N, H * da)
    Ymu, Ysd = Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-6
    m = MLP(emb.shape[1], 512, H * da, depth=2).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt, Yt = torch.tensor(emb_n, device="cuda"), torch.tensor((Y - Ymu) / Ysd, device="cuda")
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    tr, va = perm[:int(N * 0.9)], perm[int(N * 0.9):]
    for step in range(2500):
        m.train()
        idx = tr[torch.randint(0, len(tr), (256,))]
        loss = ((m(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                print(f"action MLP step {step}: train {loss.item():.4f} val {((m(Xt[va]) - Yt[va]) ** 2).mean().item():.4f}")
    m.eval()
    with torch.no_grad():
        act_pred = m(Xt).cpu().numpy().astype(np.float32)
    torch.save({"state_dict": m.state_dict(), "config": {"d_in": emb.shape[1], "hidden": 512, "d_out": H * da, "depth": 2},
                "Ymu": Ymu, "Ysd": Ysd}, f"{args.out_dir}/bridge_action_mlp.pt")

    ev = bridge_events(chunks, episode_index, frame_index)
    np.savez(f"{args.out_dir}/bridge_espr_library.npz", emb=emb_n, act_pred=act_pred, chunk=chunks,
             episode_index=episode_index, frame_index=frame_index, task=tasks, event_starts=ev)
    print(f"saved libraries to {args.out_dir}: {N} frames, {int(ev.sum())} event starts")


if __name__ == "__main__":
    main()
