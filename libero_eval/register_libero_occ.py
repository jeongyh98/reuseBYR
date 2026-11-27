"""Register the LIBERO-Occ suites libero_{spatial,object,goal,10}_occluded (scene-induced
occlusion: a white cabinet is placed between the agentview camera and the target object).
Requires the LIBERO-Occ *_occluded .bddl and .pruned_init files in the LIBERO data folders.
Imported by eval_libero.py before the environments are created."""
import os, glob
from libero.libero import benchmark as B
from libero.libero import get_libero_path

OCC_SUITES = ["libero_spatial_occluded", "libero_goal_occluded", "libero_object_occluded", "libero_10_occluded"]
_bddl_root = get_libero_path("bddl_files")

def _register():
    for suite in OCC_SUITES:
        folder = os.path.join(_bddl_root, suite)
        tasks = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(folder, "*.bddl")))
        if not tasks:
            print(f"[occ] WARN no bddl for {suite}"); continue
        B.libero_task_map[suite] = tasks
        B.task_maps[suite] = {}
        for task in tasks:
            lang = B.grab_language_from_filename(suite, task + ".bddl")
            B.task_maps[suite][task] = B.Task(
                name=task, language=lang, problem="Libero", problem_folder=suite,
                bddl_file=f"{task}.bddl", init_states_file=f"{task}.pruned_init")
        B.task_order_dict[suite] = [list(range(len(tasks)))]  # order index 0 = identity

        def _mk(nm):
            class _C(B.Benchmark):
                def __init__(self, task_order_index=0):
                    super().__init__(task_order_index=task_order_index)
                    self.name = nm
                    self._make_benchmark()
                def get_task_init_states(self, i):
                    import torch
                    p = os.path.join(get_libero_path("init_states"),
                                     self.tasks[i].problem_folder, self.tasks[i].init_states_file)
                    return torch.load(p)
            _C.__name__ = nm
            return _C
        B.register_benchmark(_mk(suite))
    print(f"[occ] registered: {[s for s in OCC_SUITES if s in B.BENCHMARK_MAPPING]}")

_register()
