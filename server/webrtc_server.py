from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import websockets
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from aiortc.rtcicetransport import RTCIceCandidate
from aiortc.rtcpeerconnection import RTCIceServer
from aiortc import MediaStreamTrack
from av import VideoFrame

from deeplive.faceswap import DetectConfig, FaceSwapEngine, has_cuda_provider


class FaceSwapVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source_track: MediaStreamTrack, engine: FaceSwapEngine, max_fps: float = 0.0):
        super().__init__()
        self._track = source_track
        self._engine = engine
        self._max_fps = float(max_fps)
        self._min_dt = 0.0 if self._max_fps <= 0 else 1.0 / self._max_fps
        self._last_proc_t = 0.0
        self._last_out: Optional[VideoFrame] = None

    async def recv(self) -> VideoFrame:
        frame: VideoFrame = await self._track.recv()
        now = time.perf_counter()

        if self._min_dt > 0 and self._last_out is not None:
            if (now - self._last_proc_t) < self._min_dt:
                # Keep timing metadata consistent with incoming stream.
                out = self._last_out
                out.pts = frame.pts
                out.time_base = frame.time_base
                return out

        img = frame.to_ndarray(format="bgr24")
        swapped = self._engine.process_frame(img)
        out = VideoFrame.from_ndarray(swapped, format="bgr24")
        out.pts = frame.pts
        out.time_base = frame.time_base

        self._last_proc_t = now
        self._last_out = out
        return out


def parse_args():
    p = argparse.ArgumentParser(description="GPU face-swap WebRTC server (aiortc)")
    p.add_argument("--signal", default="ws://127.0.0.1:8765", help="Signaling WS url")
    p.add_argument("--room", default="default", help="Room id")
    p.add_argument("--peer-id", default="gpu-server", help="Peer id label")
    p.add_argument("--source", required=True, help="Source identity image path")
    p.add_argument("--model", default="inswapper_128.onnx", help="inswapper model name/path")
    p.add_argument("--det-size", type=int, default=320)
    p.add_argument("--proc-scale", type=float, default=0.5)
    p.add_argument("--max-fps", type=float, default=0.0, help="Cap processing FPS (0=off)")
    p.add_argument("--stun", default="stun:stun.l.google.com:19302", help="STUN server url")
    return p.parse_args()


def _ice_config(stun_url: str) -> RTCConfiguration:
    return RTCConfiguration(iceServers=[RTCIceServer(urls=[stun_url])])


async def _ws_send(ws, msg: Dict[str, Any]) -> None:
    await ws.send(json.dumps(msg))


async def run_server():
    args = parse_args()

    use_gpu = has_cuda_provider()
    ctx_id = 0 if use_gpu else -1
    print(f"[*] FaceSwapEngine providers: {'CUDA+CPU' if use_gpu else 'CPU'}")

    engine = FaceSwapEngine(
        source_image=Path(args.source),
        model=args.model,
        detect=DetectConfig(det_size=args.det_size, proc_scale=args.proc_scale),
        ctx_id=ctx_id,
    )

    pc = RTCPeerConnection(configuration=_ice_config(args.stun))

    # Make sure we don’t accidentally “consume” media without using it.
    sink = MediaBlackhole()

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        print(f"[*] Track received: kind={track.kind}")
        if track.kind == "video":
            out_track = FaceSwapVideoTrack(track, engine=engine, max_fps=args.max_fps)
            pc.addTrack(out_track)
        else:
            sink.addTrack(track)

        @track.on("ended")
        async def on_ended():
            print("[*] Track ended")

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

        print(f"[*] Connected to signaling: {args.signal} (room={args.room})")

        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "signal":
                payload = msg.get("payload") or {}
            else:
                payload = msg

            ptype = payload.get("type")
            if ptype == "offer":
                offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await _ws_send(ws, {"type": "answer", "sdp": pc.localDescription.sdp})
                print("[*] Answer sent")
            elif ptype == "ice":
                c = payload.get("candidate") or {}
                if "candidate" in c:
                    cand = RTCIceCandidate(
                        sdpMid=c.get("sdpMid"),
                        sdpMLineIndex=c.get("sdpMLineIndex"),
                        candidate=c.get("candidate"),
                    )
                    await pc.addIceCandidate(cand)
            elif ptype == "bye":
                break

    await pc.close()


def main():
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

