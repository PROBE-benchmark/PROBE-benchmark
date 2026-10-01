import json
import unittest
from pathlib import Path
from laketrace.scoring.geometry import boundary_support_error

ROOT = Path(__file__).resolve().parents[1]

class ContractTests(unittest.TestCase):
    def test_same_surface_point(self):
        geometry = {"type":"axis_aligned_faces", "faces":[
            {"bounds":{"x":[0,1],"y":[0,1],"z":[10,10]}},
            {"bounds":{"x":[20,21],"y":[0,1],"z":[0,0]}}]}
        h, v = boundary_support_error((0,0,0),geometry,{"horizontal_m":7.0,"vertical_m":1.5})
        self.assertFalse(h <= 7 and v <= 1.5)

    def test_threshold_is_not_three_dimensional(self):
        geometry={"type":"axis_aligned_faces","faces":[{"bounds":{"x":[0,1],"y":[0,1],"z":[0,0]}}]}
        h,v=boundary_support_error((8,0,1.5),geometry,{"horizontal_m":7.0,"vertical_m":1.5})
        self.assertEqual((h,v),(7.0,1.5))

    def test_task_counts(self):
        path=ROOT/'data/three_waterbody_candidate_v22/truth/exam_truth.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        self.assertEqual(len(rows),48)
        self.assertEqual(sum(bool(x['source_present']) for x in rows),36)

if __name__ == '__main__':
    unittest.main()
