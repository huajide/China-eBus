from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_DATA_ROOT = PROJECT_ROOT.parent / "data"
LOCAL_INPUT_ROOT = SHARED_DATA_ROOT / "input"
BASE_VS_PARKING_ROOT = LOCAL_INPUT_ROOT / "vs_parking_nodeid_original"
VS_PARKING_ROOT = LOCAL_INPUT_ROOT / "vs_parking_nodeid"
