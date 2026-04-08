"""
Live Face Swap using insightface + inswapper_128.onnx  (fully optimised)

Architecture:
  Thread 1 (Capture)   — reads webcam into a shared holder
  Thread 2 (Detection) — runs face detection asynchronously, caches results
  Main     (Swap+Draw) — reads latest frame + cached faces, swaps, displays

Press 'q' to quit.
"""

import sys
import platform
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_IMAGE = "face.jpg"
MODEL_NAME = "inswapper_128.onnx"
CAMERA_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
DET_SIZE = 320
PROC_SCALE = 0.5


# ---------------------------------------------------------------------------
# Provider / session helpers
# ---------------------------------------------------------------------------

def _has_provider(name: str) -> bool:
    return name in onnxruntime.get_available_providers()


def _get_providers():
    providers = []
    if _has_provider("CUDAExecutionProvider"):
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _ort_session_options():
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return opts


# ---------------------------------------------------------------------------
# Model loading — only detection + recognition (skip age/gender/landmark)
# ---------------------------------------------------------------------------

def build_face_analyser(ctx_id: int):
    providers = []
    if _has_provider("CUDAExecutionProvider"):
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(DET_SIZE, DET_SIZE))
    return app


def load_swapper():
    swapper = insightface.model_zoo.get_model(
        MODEL_NAME, download=True, download_zip=True,
    )
    swapper.session = onnxruntime.InferenceSession(
        swapper.model_file,
        sess_options=_ort_session_options(),
        providers=_get_providers(),
    )
    return swapper


# ---------------------------------------------------------------------------
# FastSwapper — pre-computed latent + ONNX IO Binding on GPU
# ---------------------------------------------------------------------------

class FastSwapper:
    """Wraps the raw inswapper with three speed-ups:

    1. Source latent (embedding x emap) computed once at init.
    2. On GPU the latent lives permanently in device memory via OrtValue,
       and inference uses IO Binding to avoid per-call allocations.
    3. Paste-back logic is inlined to avoid the overhead of the generic
       swapper.get() call path.
    """

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

    # ----- IO Binding (GPU only) ---------------------------------------------

    def _setup_io_binding(self):
        current = self._session.get_providers()
        on_gpu = any(
            p in current
            for p in ("CUDAExecutionProvider",)
        )
        if not on_gpu:
            return
        try:
            self._latent_gpu = onnxruntime.OrtValue.ortvalue_from_numpy(
                self._latent_np, "cuda", 0,
            )
            self._io_binding = self._session.io_binding()
            self._io_binding.bind_ortvalue_input(
                self._input_names[1], self._latent_gpu,
            )

            # Pre-allocate output on GPU to avoid per-frame GPU reallocations.
            out_h, out_w = self._input_size
            out_np = np.empty((1, 3, out_h, out_w), dtype=np.float32)
            self._output_gpu = onnxruntime.OrtValue.ortvalue_from_numpy(
                out_np, "cuda", 0,
            )
            self._io_binding.bind_ortvalue_output(
                self._output_names[0], self._output_gpu,
            )
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
            {
                self._input_names[0]: blob,
                self._input_names[1]: self._latent_np,
            },
        )[0]

    # ----- Public API --------------------------------------------------------

    def swap(self, frame: np.ndarray, target_face) -> np.ndarray:
        aimg, M = face_align.norm_crop2(
            frame, target_face.kps, self._input_size[0],
        )
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

    # ----- Paste-back with blended mask (inlined from inswapper.py) ----------

    @staticmethod
    def _paste_back(target_img, aimg, bgr_fake, M):
        fake_diff = np.abs(
            bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        ).mean(axis=2)
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

        fake_diff = cv2.dilate(
            fake_diff, np.ones((2, 2), np.uint8), iterations=1,
        )

        k = max(mask_size // 20, 5)
        blur = tuple(2 * i + 1 for i in (k, k))
        img_mask = cv2.GaussianBlur(img_mask, blur, 0)
        fake_diff = cv2.GaussianBlur(fake_diff, (11, 11), 0)

        img_mask /= 255
        img_mask = img_mask[:, :, np.newaxis]

        merged = (
            img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        )
        return merged.astype(np.uint8)


# ---------------------------------------------------------------------------
# Thread-safe single-slot holder
# ---------------------------------------------------------------------------

class AtomicHolder:
    __slots__ = ("_value", "_lock")

    def __init__(self):
        self._value = None
        self._lock = threading.Lock()

    def put(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


# ---------------------------------------------------------------------------
# Thread 1 — Capture (DSHOW on Windows for lower latency)
# ---------------------------------------------------------------------------

class CaptureThread:
    def __init__(self, src: int, holder: AtomicHolder):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._holder = holder
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                self._holder.put(cv2.flip(frame, 1))

    def stop(self):
        self._running = False
        self.cap.release()


# ---------------------------------------------------------------------------
# Thread 2 — Async face detection
# ---------------------------------------------------------------------------

class DetectionThread:
    def __init__(
        self,
        analyser,
        frame_holder: AtomicHolder,
        packet_holder: AtomicHolder,
    ):
        self._analyser = analyser
        self._frames = frame_holder
        self._packet = packet_holder
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        inv = 1.0 / PROC_SCALE
        while self._running:
            frame = self._frames.get()
            if frame is None:
                time.sleep(0.001)
                continue

            small = cv2.resize(
                frame, None,
                fx=PROC_SCALE, fy=PROC_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )
            faces = self._analyser.get(small)
            for f in faces:
                # Avoid in-place writes: insightface Face attrs can be read-only.
                f["bbox"] = f.bbox * inv
                f["kps"] = f.kps * inv
            faces.sort(key=lambda f: f.bbox[0])

            # Keep frame + faces in sync (same exact frame the faces were found on).
            self._packet.put((frame, faces))

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Main — swap + display (runs on the main thread)
# ---------------------------------------------------------------------------

def main():
    use_gpu = _has_provider("CUDAExecutionProvider")
    ctx_id = 0 if use_gpu else -1

    prov_tags = []
    if use_gpu:
        prov_tags.append("CUDA")
    prov_tags.append("CPU")
    print(f"[*] Providers: {' > '.join(prov_tags)}")

    print("[*] Loading face analyser (detection + recognition only) …")
    analyser = build_face_analyser(ctx_id)

    print("[*] Loading inswapper model …")
    swapper_raw = load_swapper()

    # --- Source face (full analysis, run once) --------------------------------
    source_path = Path(__file__).parent / SOURCE_IMAGE
    source_img = cv2.imread(str(source_path))
    if source_img is None:
        sys.exit(
            f"[!] Cannot read source image.\n"
            f"    Put a clear face photo at: {source_path}"
        )

    source_faces = analyser.get(source_img)
    if not source_faces:
        sys.exit("[!] No face detected in the source image.")
    source_face = sorted(source_faces, key=lambda f: f.bbox[0])[0]

    fast = FastSwapper(swapper_raw, source_face)
    print(f"[+] Source latent pre-computed from {source_path.name}")
    if fast._io_binding is not None:
        print("[+] ONNX IO Binding active (latent pinned to GPU)")

    # --- Shared state ---------------------------------------------------------
    frame_holder = AtomicHolder()
    packet_holder = AtomicHolder()

    # --- Start threads --------------------------------------------------------
    capture = CaptureThread(CAMERA_INDEX, frame_holder)
    if not capture.cap.isOpened():
        sys.exit(f"[!] Cannot open camera {CAMERA_INDEX}")
    capture.start()

    while frame_holder.get() is None:
        time.sleep(0.01)

    detector = DetectionThread(analyser, frame_holder, packet_holder)
    detector.start()

    print(f"[+] Camera {CAMERA_INDEX} @ {CAM_WIDTH}x{CAM_HEIGHT} (DSHOW)"
          if platform.system() == "Windows"
          else f"[+] Camera {CAMERA_INDEX} @ {CAM_WIDTH}x{CAM_HEIGHT}")
    print("[+] Live — press 'q' to quit\n")

    fps_smooth = 0.0
    last_tick = time.perf_counter()

    # --- Swap + display loop (main thread) ------------------------------------
    while True:
        packet = packet_holder.get()
        if packet is not None:
            frame, faces = packet
            if faces:
                result = frame.copy()
                for face in faces:
                    result = fast.swap(result, face)
            else:
                result = frame
        else:
            # No detection packet yet: show latest raw frame.
            frame = frame_holder.get()
            if frame is None:
                continue
            result = frame

        cv2.putText(
            result, f"FPS: {fps_smooth:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("Live Face Swap", result)

        key = cv2.waitKey(1) & 0xFF

        # FPS based on displayed frames (includes imshow/waitKey throttling).
        now = time.perf_counter()
        dt = now - last_tick
        last_tick = now
        inst_fps = 1.0 / max(dt, 1e-9)
        fps_smooth = 0.1 * inst_fps + 0.9 * fps_smooth

        if key == ord("q"):
            break

    detector.stop()
    capture.stop()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()
