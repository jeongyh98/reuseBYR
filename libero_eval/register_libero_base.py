"""Register libero_{spatial,object,goal,10}_base: the 10 original, unperturbed LIBERO tasks per
suite. In the LIBERO-Plus package the plain suite names map to perturbed task variants, so the
canonical tasks are registered under separate names. Task names are taken from the LIBERO-Occ
*_occluded folders (same 10 tasks) and point to the plain .bddl / .pruned_init files of the
original suite. Imported by eval_libero.py before the environments are created."""
import os, glob
from libero.libero import benchmark as B
from libero.libero import get_libero_path
_bddl = get_libero_path("bddl_files")
PAIRS = [("libero_spatial_base", "libero_spatial"), ("libero_object_base", "libero_object"),
         ("libero_goal_base", "libero_goal"), ("libero_10_base", "libero_10")]

def _register():
    reg = []
    for base_name, src in PAIRS:
        occ_folder = os.path.join(_bddl, f"{src}_occluded")
        names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(occ_folder, "*.bddl")))
        names = [n for n in names if os.path.exists(os.path.join(_bddl, src, f"{n}.bddl"))]
        if not names:
            print(f"[base] WARN no plain bddl for {base_name}"); continue
        B.libero_task_map[base_name] = names
        B.task_maps[base_name] = {}
        for nm in names:
            lang = B.grab_language_from_filename(src, nm + ".bddl")
            B.task_maps[base_name][nm] = B.Task(name=nm, language=lang, problem="Libero",
                problem_folder=src, bddl_file=f"{nm}.bddl", init_states_file=f"{nm}.pruned_init")
        B.task_order_dict[base_name] = [list(range(len(names)))]
        def _mk(nm, folder):
            class _C(B.Benchmark):
                def __init__(s, task_order_index=0):
                    super().__init__(task_order_index=task_order_index); s.name = nm; s._make_benchmark()
                def get_task_init_states(s, i):
                    import torch
                    p = os.path.join(get_libero_path("init_states"), s.tasks[i].problem_folder, s.tasks[i].init_states_file)
                    return torch.load(p)
            _C.__name__ = nm; return _C
        B.register_benchmark(_mk(base_name, src)); reg.append(base_name)
    print(f"[base] registered: {reg}")

_register()
