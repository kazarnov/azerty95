from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def has_cuda_provider() -> bool:
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def ort_session_options() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def ort_providers(prefer_cuda: bool = True) -> Sequence[str]:
    providers: list[str] = []
    if prefer_cuda and has_cuda_provider():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def build_face_analyser(det_size: int = 320, ctx_id: Optional[int] = None) -> FaceAnalysis:
    """
    Build the InsightFace analyser using only detection+recognition modules.

    ctx_id:
      - 0 for GPU (if available)
      - -1 for CPU
      - None -> auto (GPU if available else CPU)
    """
    if ctx_id is None:
        ctx_id = 0 if has_cuda_provider() else -1
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=list(ort_providers(prefer_cuda=True)),
    )
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def load_swapper(model: str = "inswapper_128.onnx"):
    swapper = insightface.model_zoo.get_model(model, download=True, download_zip=True)
    swapper.session = onnxruntime.InferenceSession(
        swapper.model_file,
        sess_options=ort_session_options(),
        providers=list(ort_providers(prefer_cuda=True)),
    )
    return swapper


class FastSwapper:
    """Optimized inswapper wrapper (latent precompute + optional ORT IO binding)."""

    def __init__(self, swapper, source_face):
        self._session = swapper.session
        self._input_names = swapper.input_names
        self._output_names = swapper.output_names
        self._input_size = swapper.input_size
        self._input_mean = swapper.input_mean
        self._input_std = swapper.input_std

        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, swapper.emap)
        latent /= np.linalg.norm(latent)
        self._latent_np = latent.astype(np.float32)

        self._io_binding = None
        self._latent_gpu = None
        self._output_gpu = None
        self._setup_io_binding()

    def _setup_io_binding(self) -> None:
        current = self._session.get_providers()
        on_gpu = "CUDAExecutionProvider" in current
        if not on_gpu:
            return
        try:
            self._latent_gpu = onnxruntime.OrtValue.ortvalue_from_numpy(
                self._latent_np, "cuda", 0
            )
            self._io_binding = self._session.io_binding()
            self._io_binding.bind_ortvalue_input(self._input_names[1], self._latent_gpu)

            out_h, out_w = self._input_size
            out_np = np.empty((1, 3, out_h, out_w), dtype=np.float32)
            self._output_gpu = onnxruntime.OrtValue.ortvalue_from_numpy(out_np, "cuda", 0)
            self._io_binding.bind_ortvalue_output(self._output_names[0], self._output_gpu)
        except Exception:
            self._io_binding = None
            self._latent_gpu = None
            self._output_gpu = None

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._io_binding is not None:
            self._io_binding.bind_cpu_input(self._input_names[0], blob)
            self._session.run_with_iobinding(self._io_binding)
            return self._io_binding.copy_outputs_to_cpu()[0]
        return self._session.run(
            self._output_names,
            {self._input_names[0]: blob, self._input_names[1]: self._latent_np},
        )[0]

    def swap(self, frame: np.ndarray, target_face) -> np.ndarray:
        aimg, M = face_align.norm_crop2(frame, target_face.kps, self._input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self._input_std,
            self._input_size,
            (self._input_mean, self._input_mean, self._input_mean),
            swapRB=True,
        )
        pred = self._infer(blob)

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return self._paste_back(frame, aimg, bgr_fake, M)

    @staticmethod
    def _paste_back(target_img, aimg, bgr_fake, M):
        fake_diff = np.abs(bgr_fake.astype(np.float32) - aimg.astype(np.float32)).mean(
            axis=2
        )
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0

        IM = cv2.invertAffineTransform(M)
        h, w = target_img.shape[:2]
        ah, aw = aimg.shape[:2]

        img_white = np.full((ah, aw), 255, dtype=np.float32)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0.0)
        img_white = cv2.warpAffine(img_white, IM, (w, h), borderValue=0.0)
        fake_diff = cv2.warpAffine(fake_diff, IM, (w, h), borderValue=0.0)

        img_white[img_white > 20] = 255
        fake_diff[fake_diff < 10] = 0
        fake_diff[fake_diff >= 10] = 255

        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))

        k = max(mask_size // 10, 10)
        img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)

        fake_diff = cv2.dilate(fake_diff, np.ones((2, 2), np.uint8), iterations=1)

        k = max(mask_size // 20, 5)
        blur = tuple(2 * i + 1 for i in (k, k))
        img_mask = cv2.GaussianBlur(img_mask, blur, 0)
        fake_diff = cv2.GaussianBlur(fake_diff, (11, 11), 0)

        img_mask /= 255
        img_mask = img_mask[:, :, np.newaxis]

        merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        return merged.astype(np.uint8)


@dataclass(frozen=True)
class DetectConfig:
    det_size: int = 320
    proc_scale: float = 0.5


class FaceSwapEngine:
    def __init__(
        self,
        source_image: str | Path,
        model: str = "inswapper_128.onnx",
        detect: DetectConfig = DetectConfig(),
        ctx_id: Optional[int] = None,
    ):
        self.detect_cfg = detect
        self.analyser = build_face_analyser(det_size=detect.det_size, ctx_id=ctx_id)
        self.swapper_raw = load_swapper(model=model)

        source_path = Path(source_image)
        source_img = cv2.imread(str(source_path))
        if source_img is None:
            raise FileNotFoundError(f"Cannot read source image: {source_path}")
        source_faces = self.analyser.get(source_img)
        if not source_faces:
            raise RuntimeError("No face detected in the source image.")
        source_face = sorted(source_faces, key=lambda f: f.bbox[0])[0]

        self.fast = FastSwapper(self.swapper_raw, source_face)

    def detect_faces(self, frame_bgr: np.ndarray):
        inv = 1.0 / max(self.detect_cfg.proc_scale, 1e-6)
        small = cv2.resize(
            frame_bgr,
            None,
            fx=self.detect_cfg.proc_scale,
            fy=self.detect_cfg.proc_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        faces = self.analyser.get(small)
        for f in faces:
            f["bbox"] = f.bbox * inv
            f["kps"] = f.kps * inv
        faces.sort(key=lambda f: f.bbox[0])
        return faces

    def swap_faces(self, frame_bgr: np.ndarray, faces: Iterable) -> np.ndarray:
        result = frame_bgr
        for face in faces:
            result = self.fast.swap(result, face)
        return result

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        faces = self.detect_faces(frame_bgr)
        if not faces:
            return frame_bgr
        return self.swap_faces(frame_bgr.copy(), faces)

