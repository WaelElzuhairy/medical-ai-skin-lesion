"""
Unified image loader for PNG, JPEG, and DICOM inputs.

Dispatches by file extension and returns a `LoadedImage` so downstream code
(preprocessing, inference, Grad-CAM, UI) only ever sees a PIL.Image plus a
metadata dict — it never has to branch on format. DICOM pixel data is
windowed via VOI LUT when present and min-max normalized to 8-bit RGB so the
same EfficientNet-B4 input pipeline applies to all modalities.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

DICOM_EXTS = {".dcm", ".dicom"}
PIL_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class LoadedImage:
    image: Image.Image
    modality: str  # "JPEG" | "PNG" | "DICOM"
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_pil(path: Path) -> LoadedImage:
    img = Image.open(path).convert("RGB")
    return LoadedImage(
        image=img,
        modality=path.suffix.lstrip(".").upper(),
        metadata={"filename": path.name, "size": img.size},
    )


def _load_dicom(path_or_bytes: Path | bytes) -> LoadedImage:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        ds = pydicom.dcmread(io.BytesIO(path_or_bytes))
        filename = "<bytes>"
    else:
        ds = pydicom.dcmread(str(path_or_bytes))
        filename = path_or_bytes.name

    arr = ds.pixel_array
    # Apply VOI LUT (windowing) if the DICOM specifies one — required for
    # CT/MR to look correct; harmless for already-windowed modalities.
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = arr.astype(np.float32)
    # Photometric interpretation: invert if MONOCHROME1 (white-on-black convention).
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    arr = (arr * 255.0).astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        img = Image.fromarray(arr[..., :3], mode="RGB")
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")

    metadata = {
        "filename": filename,
        "modality": getattr(ds, "Modality", None),
        "patient_age": getattr(ds, "PatientAge", None),
        "patient_sex": getattr(ds, "PatientSex", None),
        "body_part": getattr(ds, "BodyPartExamined", None),
        "study_date": getattr(ds, "StudyDate", None),
        "size": img.size,
    }
    return LoadedImage(image=img, modality="DICOM", metadata=metadata)


def load_image(path: str | Path) -> LoadedImage:
    """Load an image from disk. Dispatches on file extension."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext in PIL_EXTS:
        return _load_pil(path)
    if ext in DICOM_EXTS:
        return _load_dicom(path)
    raise ValueError(f"Unsupported image format: {ext!r}")


def load_from_bytes(data: bytes, filename: str) -> LoadedImage:
    """Load an image from an in-memory byte buffer (used by Streamlit upload)."""
    ext = Path(filename).suffix.lower()
    if ext in PIL_EXTS:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return LoadedImage(
            image=img,
            modality=ext.lstrip(".").upper(),
            metadata={"filename": filename, "size": img.size},
        )
    if ext in DICOM_EXTS:
        return _load_dicom(bytes(data))
    raise ValueError(f"Unsupported image format: {ext!r}")
