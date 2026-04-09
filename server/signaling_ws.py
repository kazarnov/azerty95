from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import websockets
from websockets.server import WebSocketServerProtocol


@dataclass
class Peer:
    ws: WebSocketServerProtocol
    peer_id: str


@dataclass
class Room:
    peers: Dict[str, Peer] = field(default_factory=dict)

    def other(self, peer_id: str) -> Optional[Peer]:
        for pid, p in self.peers.items():
            if pid != peer_id:
                return p
        return None


class SignalingServer:
    def __init__(self):
        self._rooms: Dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def _send(self, ws: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
        await ws.send(json.dumps(payload))

    async def handler(self, ws: WebSocketServerProtocol):
        room_id: Optional[str] = None
        peer_id: Optional[str] = None
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    await self._send(ws, {"type": "error", "error": "invalid_json"})
                    continue

                mtype = msg.get("type")
                if mtype == "join":
                    room_id = str(msg.get("room") or "default")
                    peer_id = str(msg.get("peer_id") or secrets.token_hex(4))
                    async with self._lock:
                        room = self._rooms.setdefault(room_id, Room())
                        if peer_id in room.peers:
                            peer_id = f"{peer_id}-{secrets.token_hex(2)}"
                        room.peers[peer_id] = Peer(ws=ws, peer_id=peer_id)
                        size = len(room.peers)

                    await self._send(
                        ws,
                        {
                            "type": "joined",
                            "room": room_id,
                            "peer_id": peer_id,
                            "room_size": size,
                        },
                    )

                    # Notify the other peer (if present).
                    async with self._lock:
                        other = self._rooms.get(room_id, Room()).other(peer_id)
                    if other is not None:
                        await self._send(other.ws, {"type": "peer_joined", "peer_id": peer_id})
                    continue

                if room_id is None or peer_id is None:
                    await self._send(ws, {"type": "error", "error": "must_join_first"})
                    continue

                # Relay any signaling payload to the other peer in the room.
                async with self._lock:
                    room = self._rooms.get(room_id)
                    other = None if room is None else room.other(peer_id)

                if other is None:
                    await self._send(ws, {"type": "error", "error": "no_peer_in_room"})
                    continue

                await self._send(
                    other.ws,
                    {
                        "type": "signal",
                        "from": peer_id,
                        "payload": msg,
                    },
                )
        finally:
            if room_id is None or peer_id is None:
                return
            async with self._lock:
                room = self._rooms.get(room_id)
                if room is None:
                    return
                room.peers.pop(peer_id, None)
                if not room.peers:
                    self._rooms.pop(room_id, None)


def parse_args():
    p = argparse.ArgumentParser(description="Minimal WS signaling server (1:1 rooms)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


async def amain():
    args = parse_args()
    server = SignalingServer()
    async with websockets.serve(server.handler, args.host, args.port):
        print(f"[*] Signaling WS listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

