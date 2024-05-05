from pathlib import Path

SOURCES_ROOT = Path(__file__).parent
PROJECT_ROOT = SOURCES_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
NEPTUNE_CONFIG_PATH = PROJECT_ROOT / "neptune.cfg"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
