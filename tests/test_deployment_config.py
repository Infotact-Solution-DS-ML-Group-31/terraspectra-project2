"""Unit tests for TerraSpectra containerization and deployment configuration."""

from __future__ import annotations

from pathlib import Path
import unittest


class TestDeploymentConfig(unittest.TestCase):
    """Test suite validating deployment configuration files, Dockerfile, and scripts."""

    @classmethod
    def setUpClass(cls):
        """Locate root directory relative to test file."""
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.dockerfile_path = cls.root_dir / "Dockerfile"
        cls.dockerignore_path = cls.root_dir / ".dockerignore"
        cls.requirements_path = cls.root_dir / "requirements.txt"
        cls.script_path = cls.root_dir / "scripts" / "run_local.ps1"

    def test_configuration_files_exist(self):
        """Verify Dockerfile, .dockerignore, requirements.txt, and scripts/run_local.ps1 exist."""
        self.assertTrue(self.dockerfile_path.is_file(), "Dockerfile must exist at project root.")
        self.assertTrue(self.dockerignore_path.is_file(), ".dockerignore must exist at project root.")
        self.assertTrue(self.requirements_path.is_file(), "requirements.txt must exist at project root.")
        self.assertTrue(self.script_path.is_file(), "scripts/run_local.ps1 must exist.")

    def test_dockerfile_contents(self):
        """Verify Dockerfile exposes ports 8000 and 8501, and references src.api.main:app and requirements.txt."""
        content = self.dockerfile_path.read_text(encoding="utf-8")

        # Base image
        self.assertIn("FROM python:3.11-slim", content)

        # Port exposures
        self.assertIn("EXPOSE 8000", content, "Dockerfile must expose port 8000 for FastAPI.")
        self.assertIn("EXPOSE 8501", content, "Dockerfile must expose port 8501 for Streamlit.")

        # References
        self.assertIn("requirements.txt", content, "Dockerfile must reference requirements.txt.")
        self.assertIn("src.api.main:app", content, "Dockerfile must reference src.api.main:app.")

        # Best practice environment variables
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", content)
        self.assertIn("PYTHONUNBUFFERED=1", content)

    def test_dockerignore_contents(self):
        """Verify .dockerignore excludes critical development and cache artifacts."""
        content = self.dockerignore_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        expected_entries = [
            ".venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".git/",
            "*.pyc",
            "checkpoints/*.pt",
            ".DS_Store",
        ]
        for entry in expected_entries:
            self.assertIn(entry, lines, f".dockerignore must contain '{entry}'.")

    def test_requirements_contains_required_dependencies(self):
        """Verify requirements.txt contains all specified foundational packages."""
        content = self.requirements_path.read_text(encoding="utf-8").lower()

        required_packages = [
            "torch",
            "fastapi",
            "uvicorn",
            "streamlit",
            "pydeck",
            "numpy",
            "scikit-learn",
        ]
        for pkg in required_packages:
            self.assertTrue(
                any(pkg in line for line in content.splitlines()),
                f"requirements.txt must contain '{pkg}' dependency.",
            )

    def test_run_local_script_contents(self):
        """Verify scripts/run_local.ps1 references virtualenv, modes, uvicorn, and streamlit."""
        content = self.script_path.read_text(encoding="utf-8")

        self.assertIn("Activate.ps1", content, "run_local.ps1 must reference virtual environment activation.")
        self.assertIn("dashboard", content, "run_local.ps1 must support 'dashboard' mode.")
        self.assertIn("api", content, "run_local.ps1 must support 'api' mode.")
        self.assertIn("src.api.main:app", content, "run_local.ps1 must reference FastAPI app.")
        self.assertIn("app.py", content, "run_local.ps1 must reference Streamlit app.py.")


if __name__ == "__main__":
    unittest.main()
