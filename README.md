# Live Face Swap with inswapper_128

Real-time webcam face swap powered by [insightface](https://github.com/deepinsight/insightface) and the `inswapper_128.onnx` model.

## Pipeline overview

```
Source image ──►┐
                ├─► Face Analyser (buffalo_l) ─► inswapper_128 ─► Blended frame ─► Display
Webcam frame ──►┘         │                            │
                     detect faces               swap each face
                     + extract embeddings       with source identity
```

1. **Face detection & analysis** — `buffalo_l` model detects faces and extracts 512-d ArcFace embeddings.
2. **Embedding transform** — the source embedding is projected through the model's internal embedding map.
3. **Face generation** — the ONNX model produces a 128×128 swapped face.
4. **Paste-back & blending** — an affine warp + Gaussian-blurred mask composites the result onto the original frame.

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

**GPU (recommended for real-time):**
```bash
pip install -r requirements.txt
```

**CPU only** — edit `requirements.txt` and replace `onnxruntime-gpu` with `onnxruntime`, then:
```bash
pip install -r requirements.txt
```

### 3. Download the model

**Option A — automatic:** the script auto-downloads the model on first run.

**Option B — manual:** download `inswapper_128.onnx` from
<https://huggingface.co/ezioruan/inswapper_128.onnx> and place it at:

```
~/.insightface/models/inswapper_128.onnx
```

Or pass the path with `--model path/to/inswapper_128.onnx`.

## Usage

```bash
# Basic — use your webcam and a source face photo
python main.py face.jpg

# Use GPU acceleration
python main.py face.jpg --gpu

# Use a different camera
python main.py face.jpg --camera 1

# Lower detection resolution for more speed
python main.py face.jpg --gpu --det-size 320

# Supply a custom model path
python main.py face.jpg --model ./models/inswapper_128.onnx
```

Press **q** to quit the live preview window.

## WebRTC (GPU server + Windows client + OBS Virtual Camera)

This repo also includes a simple **1:1 WebRTC** setup so you can run the heavy face-swap on a **GPU server** and use a lightweight **Windows client** that outputs the swapped stream as a **Virtual Webcam** for OBS.

### Components

- **Signaling server**: `server/signaling_ws.py` (WebSocket relay for SDP/ICE)
- **GPU WebRTC face-swap server**: `server/webrtc_server.py`
- **Windows client -> VirtualCam**: `client/client_virtualcam.py`

### Run (LAN / same network)

1) **On the GPU server**

Start signaling:

```bash
python -m server.signaling_ws --host 0.0.0.0 --port 8765
```

Start the WebRTC face-swap server (set your source face image path):

```bash
python -m server.webrtc_server --signal ws://<SERVER_IP>:8765 --room default --source face.jpg --max-fps 30
```

2) **On your Windows PC**

Run the client:

```bash
python -m client.client_virtualcam --signal ws://<SERVER_IP>:8765 --room default --camera 0 --width 640 --height 480 --fps 30
```

3) **In OBS**

- Add **Video Capture Device**
- Choose the virtual camera device created by `pyvirtualcam`

### Internet access (STUN/TURN)

WebRTC usually needs ICE help when you’re not on the same LAN.

- **STUN**: enabled by default (`stun:stun.l.google.com:19302`) and often works for many home networks.
- **TURN**: for reliable connectivity across strict NATs/firewalls, run a TURN server (e.g., **coturn**) and add it to both the server/client ICE configuration (future enhancement).

### Performance notes

- Start at **640x480** and **30 FPS**.
- Use `--max-fps` on the GPU server to cap processing and avoid backlog/latency spikes.
- If latency grows over time, reduce FPS/resolution and/or increase `--proc-scale` (detection scale) trade-offs.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `source` | *(required)* | Path to the face image whose identity you want to use |
| `--camera` | `0` | Webcam device index |
| `--model` | auto-download | Path to `inswapper_128.onnx` |
| `--det-size` | `640` | Face detection resolution (lower = faster) |
| `--max-fps` | `0` (unlimited) | Cap output frame rate |
| `--gpu` | off | Enable CUDA/GPU inference |

## Performance tips

- Use `--gpu` with `onnxruntime-gpu` installed for best throughput.
- Lower `--det-size` to `320` for faster detection at slight accuracy cost.
- Use a smaller webcam resolution if needed (the script defaults to 1280×720).
- On CPU you can expect ~2-5 FPS; on a modern GPU (RTX 3060+) you can hit 15-30+ FPS.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No face detected in the source image` | Use a clear, front-facing photo with good lighting |
| `Cannot open camera` | Check the `--camera` index; try `0`, `1`, etc. |
| ONNX Runtime CUDA errors | Make sure your CUDA/cuDNN versions match `onnxruntime-gpu` requirements |
| Slow on CPU | Switch to `--gpu` or lower `--det-size` to `320` |
