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
from pathlib import Path

import cv2
from deeplive.faceswap import FaceSwapEngine, has_cuda_provider

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


def engine_detect_cfg():
    from deeplive.faceswap import DetectConfig

    return DetectConfig(det_size=DET_SIZE, proc_scale=PROC_SCALE)


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
        engine: FaceSwapEngine,
        frame_holder: AtomicHolder,
        packet_holder: AtomicHolder,
    ):
        self._engine = engine
        self._frames = frame_holder
        self._packet = packet_holder
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self._running:
            frame = self._frames.get()
            if frame is None:
                time.sleep(0.001)
                continue
            faces = self._engine.detect_faces(frame)

            # Keep frame + faces in sync (same exact frame the faces were found on).
            self._packet.put((frame, faces))

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Main — swap + display (runs on the main thread)
# ---------------------------------------------------------------------------

def main():
    use_gpu = has_cuda_provider()

    prov_tags = []
    if use_gpu:
        prov_tags.append("CUDA")
    prov_tags.append("CPU")
    print(f"[*] Providers: {' > '.join(prov_tags)}")

    print("[*] Loading face swap engine …")
    source_path = Path(__file__).parent / SOURCE_IMAGE
    try:
        engine = FaceSwapEngine(
            source_image=source_path,
            model=MODEL_NAME,
            detect=engine_detect_cfg(),
            ctx_id=0 if use_gpu else -1,
        )
    except Exception as e:
        sys.exit(f"[!] Failed to initialize engine: {e}")

    print(f"[+] Source latent pre-computed from {source_path.name}")
    if getattr(engine.fast, "_io_binding", None) is not None:
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

    detector = DetectionThread(engine, frame_holder, packet_holder)
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
                result = engine.swap_faces(frame.copy(), faces)
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
