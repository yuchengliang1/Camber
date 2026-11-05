import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTest(unittest.TestCase):
    def test_python_sources_parse(self):
        sources = sorted(ROOT.rglob("*.py"))
        self.assertGreater(len(sources), 30)
        for path in sources:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_machine_specific_paths(self):
        patterns = [
            re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\\\]"),
            re.compile(r"/(?:home|Users)/[^/]+/", re.I),
        ]
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".json", ".md", ".txt"}:
                text = path.read_text(encoding="utf-8")
                for pattern in patterns:
                    self.assertIsNone(pattern.search(text), str(path))

    def test_no_private_recordings_or_checkpoints(self):
        self.assertFalse(any(ROOT.rglob("*.npz")))
        self.assertFalse(any(ROOT.rglob("*.pt")))
        self.assertFalse(any(ROOT.rglob("*.ckpt")))

    def test_no_transient_or_private_directories(self):
        forbidden = {"results", "local_runs", "checkpoints", "protocols", "manuscript", "tmp"}
        present = {
            path.name.lower()
            for path in ROOT.rglob("*")
            if path.is_dir() and path.name.lower() in forbidden
        }
        self.assertEqual(present, set())

    def test_local_qa_outputs_are_gitignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("__pycache__/", ignored)
        self.assertIn("_qa_figures/", ignored)
        self.assertIn("_qa_rendered/", ignored)


if __name__ == "__main__":
    unittest.main()
