from __future__ import annotations

import argparse
import asyncio
import json
import platform
import threading
import time
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pyvirtualcam
import websockets
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcicetransport import RTCIceCandidate
from aiortc.rtcpeerconnection import RTCIceServer
from aiortc import MediaStreamTrack
from av import VideoFrame


class AtomicFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class CaptureThread:
    def __init__(self, camera: int, holder: AtomicFrame, width: int, height: int):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        self.cap = cv2.VideoCapture(camera, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
                # Mirror like a typical selfie preview.
                self._holder.put(cv2.flip(frame, 1))
            else:
                time.sleep(0.001)

    def stop(self):
        self._running = False
        self.cap.release()


class WebcamVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, holder: AtomicFrame, fps: int):
        super().__init__()
        self._holder = holder
        self._fps = max(int(fps), 1)
        self._time_base = Fraction(1, self._fps)
        self._pts = 0

    async def recv(self) -> VideoFrame:
        # Wait until we have at least one frame.
        frame = self._holder.get()
        while frame is None:
            await asyncio.sleep(0.005)
            frame = self._holder.get()

        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = self._pts
        vf.time_base = self._time_base
        self._pts += 1
        await asyncio.sleep(self._time_base)  # pacing
        return vf


def parse_args():
    p = argparse.ArgumentParser(description="Windows WebRTC client -> virtual camera (OBS)")
    p.add_argument("--signal", default="ws://127.0.0.1:8765", help="Signaling WS url")
    p.add_argument("--room", default="default", help="Room id")
    p.add_argument("--peer-id", default="pc-client", help="Peer id label")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, help="Capture/WebRTC send fps")
    p.add_argument("--stun", default="stun:stun.l.google.com:19302", help="STUN server url")
    p.add_argument("--virtual-name", default="DeepLive VirtualCam", help="Virtual camera name")
    return p.parse_args()


def _ice_config(stun_url: str) -> RTCConfiguration:
    return RTCConfiguration(iceServers=[RTCIceServer(urls=[stun_url])])


async def _ws_send(ws, msg: Dict[str, Any]) -> None:
    await ws.send(json.dumps(msg))


async def run_client():
    args = parse_args()

    holder = AtomicFrame()
    capture = CaptureThread(args.camera, holder, args.width, args.height)
    if not capture.cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    capture.start()

    pc = RTCPeerConnection(configuration=_ice_config(args.stun))
    pc.addTrack(WebcamVideoTrack(holder, fps=args.fps))

    cam: Optional[pyvirtualcam.Camera] = None

    async def ensure_cam(width: int, height: int, fps: int) -> pyvirtualcam.Camera:
        nonlocal cam
        if cam is None:
            cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
                device=None,
                backend="auto",
            )
            print(f"[*] VirtualCam ready: {cam.device}")
        return cam

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        print(f"[*] Track received: kind={track.kind}")
        if track.kind != "video":
            return

        async def pump():
            while True:
                try:
                    frame: VideoFrame = await track.recv()
                except Exception:
                    break
                img = frame.to_ndarray(format="rgb24")
                vc = await ensure_cam(img.shape[1], img.shape[0], args.fps)
                vc.send(img)
                vc.sleep_until_next_frame()

        asyncio.create_task(pump())

    async with websockets.connect(args.signal) as ws:
        await _ws_send(ws, {"type": "join", "room": args.room, "peer_id": args.peer_id})

        async def on_icecandidate(candidate):
            if candidate is None:
                return
            await _ws_send(
                ws,
                {
                    "type": "ice",
                    "candidate": {
                        "candidate": candidate.to_sdp(),
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                },
            )

        pc.on("icecandidate", on_icecandidate)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await _ws_send(ws, {"type": "offer", "sdp": pc.localDescription.sdp})
        print("[*] Offer sent")

        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "signal":
                payload = msg.get("payload") or {}
            else:
                payload = msg

            ptype = payload.get("type")
            if ptype == "answer":
                answer = RTCSessionDescription(sdp=payload["sdp"], type="answer")
                await pc.setRemoteDescription(answer)
                print("[*] Answer received")
            elif ptype == "ice":
                c = payload.get("candidate") or {}
                if "candidate" in c:
                    cand = RTCIceCandidate(
                        sdpMid=c.get("sdpMid"),
                        sdpMLineIndex=c.get("sdpMLineIndex"),
                        candidate=c.get("candidate"),
                    )
                    await pc.addIceCandidate(cand)

    await pc.close()
    capture.stop()
    if cam is not None:
        cam.close()


def main():
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

