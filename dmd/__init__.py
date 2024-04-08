from pathlib import Path


SOURCES_ROOT = Path(__file__).parent
PROJECT_ROOT = SOURCES_ROOT.parent
EDM_PACKAGE_DIR = SOURCES_ROOT / 'edm'  # this is used for loading pkl (see `EDMGenerator`)
DATA_DIR = PROJECT_ROOT / 'data'
