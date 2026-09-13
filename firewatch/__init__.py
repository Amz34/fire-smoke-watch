"""Fire & smoke detection on ONNX Runtime, CPU only.

Public surface: `Detector` (inference + annotation) and `Detection` (a result).
See ../README.md for the model interface contract.
"""
from .detector import CLASS_NAMES, Detection, Detector, letterbox

__all__ = ["Detector", "Detection", "CLASS_NAMES", "letterbox"]
__version__ = "0.1.0"
