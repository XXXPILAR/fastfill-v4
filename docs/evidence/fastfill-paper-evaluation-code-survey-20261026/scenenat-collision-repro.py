"""Isolated CPU test of pinned upstream metric; no model or dataset evaluation."""
import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
base = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    raise SystemExit('Usage: python scenenat-collision-repro.py /path/to/pinned/collision_metric.py')
path = Path(sys.argv[1])
expected_sha = '4376451e2dc7ab2527566df8c10e11a26e27eb6df06bc6524e68dfead1dcc485'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit('Source hash differs from the audited SceneNAT commit; refusing to execute it.')
spec = importlib.util.spec_from_file_location('pinned_collision_metric', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import numpy as np
boxes = np.array([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], dtype=float)
result = module.get_collision_matrix(boxes, with_fig=False)
output = {
    'executed': True, 'scope': 'One isolated CPU geometry function from upstream pinned source; no model, rendering, dataset or paper reproduction',
    'input_half_extent_boxes': boxes.tolist(),
    'actual_intersection_matrix': result[0].tolist(),
    'actual_inter_vol_mean': float(result[1]),
    'actual_inter_vol_sum': float(result[2]),
    'actual_iomin': float(result[3]),
    'reference_unordered_pair_intersection_volume': 8.0,
    'reference_pair_iomin': 1.0,
    'assertions': {'pair_volume': bool(np.isclose(result[0][0,1],8)), 'reported_iomin_exceeds_one': bool(result[3] > 1)},
    'versions': {'numpy': np.__version__},
    'interpretation': 'Sizes are treated as half extents; volume denominator omits factor 8. Mean uses M*M entries including zeroed diagonal; total includes both (i,j) and (j,i).'
}
for name in ['shapely', 'matplotlib']:
    imported = __import__(name)
    output['versions'][name] = imported.__version__
(base/'scenenat-collision-repro.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output))
