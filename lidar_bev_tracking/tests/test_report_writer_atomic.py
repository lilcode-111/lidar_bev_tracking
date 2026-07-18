import tempfile
import unittest
from pathlib import Path

from bev_tracking.report_writer import atomic_write_json, atomic_write_text


class ReportWriterAtomicTest(unittest.TestCase):
    def test_atomic_write_json_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"

            atomic_write_json(path, {"ok": True})

            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "ok": true\n}')
            self.assertFalse((Path(tmp) / ".summary.json.tmp").exists())

    def test_atomic_write_text_does_not_leave_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.csv"

            atomic_write_text(path, "a,b\n1,2\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "a,b\n1,2\n")
            self.assertFalse((Path(tmp) / ".frames.csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
