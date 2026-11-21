import argparse
import asyncio
import contextlib
import logging
import os
import random
import signal
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import IO

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.loop import build_runtime
from dap.backoff import backoff_seconds
from dap.config import get_settings
from dap.db.session import make_engine, make_sessionmaker
from dap.engine import demo
from dap.engine.leases import reap_expired_leases
from dap.engine.queue import claim_next_step
from dap.engine.runner import EngineConfig, Handler, execute_claim

log = logging.getLogger(__name__)


def default_handlers() -> dict[str, Handler]:
    return {demo.CHAIN_KIND: demo.chain_step}


@dataclass
class Worker:
    worker_id: str
    sessionmaker: async_sessionmaker[AsyncSession]
    handlers: Mapping[str, Handler]
    config: EngineConfig
    rng: random.Random = field(default_factory=random.Random)
    stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self) -> None:
        log.info("worker %s started", self.worker_id)
        while not self.stop.is_set():
            try:
                busy = await self.tick()
            except Exception:
                log.exception("worker %s tick failed", self.worker_id)
                busy = False
            if not busy:
                await self.idle()
        log.info("worker %s stopped cleanly", self.worker_id)

    async def tick(self) -> bool:
        async with self.sessionmaker.begin() as session:
            await reap_expired_leases(session, self.retry_delay)
        async with self.sessionmaker.begin() as session:
            claim = await claim_next_step(session, self.worker_id, self.config.lease_seconds)
        if claim is None:
            return False
        log.info(
            "worker %s claimed step seq=%s attempt=%s", self.worker_id, claim.seq, claim.attempt
        )
        await execute_claim(self.sessionmaker, claim, self.handlers, self.config, self.rng)
        return True

    def retry_delay(self, attempt: int) -> float:
        cfg = self.config
        return backoff_seconds(attempt, cfg.retry_base_seconds, cfg.retry_cap_seconds, self.rng)

    async def idle(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.stop.wait(), self.config.poll_seconds)


def install_stop_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def handle(signum: int, frame: FrameType | None) -> None:
        if stop.is_set():
            log.warning("second interrupt: exiting immediately")
            os._exit(130)
        log.info("interrupt received: finishing current step, then exiting")
        loop.call_soon_threadsafe(stop.set)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


async def serve(worker_id: str) -> None:
    settings = get_settings()
    engine = make_engine(settings)
    sessionmaker = make_sessionmaker(engine)
    agent = await build_runtime(settings, sessionmaker)
    worker = Worker(
        worker_id=worker_id,
        sessionmaker=sessionmaker,
        handlers={**default_handlers(), **agent.handlers()},
        config=EngineConfig.from_settings(settings),
    )
    install_stop_handlers(worker.stop)
    try:
        await worker.run()
    finally:
        await engine.dispose()


def spawn_worker_process(
    worker_id: str,
    env: Mapping[str, str],
    cwd: Path,
    stdout: IO[bytes] | None = None,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "dap.engine.worker", "--worker-id", worker_id],
        env={**os.environ, **env},
        cwd=cwd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one durable step worker.")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{args.worker_id}] %(levelname)s %(message)s",
    )
    asyncio.run(serve(args.worker_id))


if __name__ == "__main__":
    main()
