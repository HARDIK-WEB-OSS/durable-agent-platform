"""python_exec: run model-written Python in a locked-down, throwaway Docker container.

Each call starts `docker run --rm` with:
    --network none                          no network at all
    --read-only --tmpfs /tmp                root filesystem read-only; only /tmp is writable
    --memory 256m --memory-swap 256m        hard memory cap, no swap
    --cpus 1 --pids-limit 64                one CPU; no fork bombs
    --cap-drop ALL                          no Linux capabilities
    --security-opt no-new-privileges        setuid binaries cannot raise privileges
    --user 65534:65534                      runs as nobody
The code goes in on stdin (python -I -), so nothing is mounted from the host. A wall-clock
timeout kills the container by name, not just the docker CLI process. Output is capped while
it is read, so a print loop cannot fill host memory.
"""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass

from dap.agent.tools.base import ToolError

DOCKER_RUN_FAILED = 125  # docker itself failed (daemon down, bad flag), not the user's code
KILLED = 137  # SIGKILL: memory limit or our timeout
CHUNK_BYTES = 4096
CLEANUP_TIMEOUT_S = 15.0  # upper bound on any docker kill/rm/inspect call


class SandboxUnavailableError(Exception):
    """Docker or the pinned image is missing: an infrastructure fault, not a tool result."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    output: str
    timed_out: bool
    output_truncated: bool


@dataclass(frozen=True)
class DockerSandbox:
    image: str
    timeout_s: float = 10.0
    memory: str = "256m"
    cpus: str = "1"
    pids_limit: int = 64
    tmpfs_size: str = "64m"
    max_output_bytes: int = 16_000
    docker: str = "docker"

    def __post_init__(self) -> None:
        if "@sha256:" not in self.image:
            raise ValueError(f"sandbox image must be pinned by digest, got {self.image!r}")

    def run_command(self, container: str) -> list[str]:
        return [
            self.docker,
            "run",
            "--rm",
            "--interactive",
            "--name",
            container,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.tmpfs_size}",
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            self.image,
            "python",
            "-I",
            "-",
        ]

    async def check_available(self) -> None:
        """Fail fast at worker startup if Docker or the pinned image is missing."""
        code, output = await self.docker_cli("image", "inspect", "--format", "{{.Id}}", self.image)
        if code != 0:
            raise SandboxUnavailableError(
                f"sandbox image {self.image} is not available ({output.strip()[:200]}). "
                f"Run: docker pull {self.image}  or set PYTHON_EXEC_ENABLED=false"
            )

    async def run(self, code: str) -> SandboxResult:
        container = f"dap-sbx-{uuid.uuid4().hex[:16]}"
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.run_command(container),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(f"docker executable {self.docker!r} not found") from exc
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(code.encode())
        proc.stdin.close()
        buffer = bytearray()
        truncated = timed_out = False
        removal: asyncio.Task[None] | None = None

        def stop() -> None:
            # Never wait on `docker rm` while the stream is unread: the daemon will not finish
            # removing a container whose output it is still trying to deliver, and we would
            # deadlock. Kill our docker client (closes the stream), remove in the background.
            nonlocal removal
            if removal is None:
                removal = asyncio.create_task(self.kill(container))
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

        # Read to EOF even after stopping: asyncio's wait() returns only once the pipe closes.
        try:
            async with asyncio.timeout(self.timeout_s):
                while chunk := await proc.stdout.read(CHUNK_BYTES):
                    if truncated:
                        continue
                    buffer += chunk
                    if len(buffer) > self.max_output_bytes:
                        truncated = True
                        stop()
                await proc.wait()
        except TimeoutError:
            timed_out = True
        finally:
            if proc.returncode is None:
                stop()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(CLEANUP_TIMEOUT_S):
                        while await proc.stdout.read(CHUNK_BYTES):
                            pass
                        await proc.wait()
            if removal is not None:
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(CLEANUP_TIMEOUT_S):
                        await removal
        exit_code = proc.returncode if proc.returncode is not None else KILLED
        output = bytes(buffer[: self.max_output_bytes]).decode(errors="replace")
        if exit_code == DOCKER_RUN_FAILED and not timed_out:
            raise SandboxUnavailableError(f"docker run failed: {output.strip()[:300]}")
        return SandboxResult(exit_code, output, timed_out, truncated)

    async def kill(self, container: str) -> None:
        await self.docker_cli("rm", "--force", container)

    async def docker_cli(self, *args: str) -> tuple[int, str]:
        """Run a short docker command; never waits longer than CLEANUP_TIMEOUT_S."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(f"docker executable {self.docker!r} not found") from exc
        try:
            async with asyncio.timeout(CLEANUP_TIMEOUT_S):
                out, _ = await proc.communicate()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return 1, f"docker {' '.join(args[:2])} timed out after {CLEANUP_TIMEOUT_S:g} s"
        return proc.returncode or 0, out.decode(errors="replace")


@dataclass(frozen=True)
class PythonExecTool:
    sandbox: DockerSandbox
    name: str = "python_exec"

    @property
    def description(self) -> str:
        return (
            "Run a Python 3.11 script (standard library only, no network, no files except "
            f"/tmp, {self.sandbox.timeout_s:g} s limit). tool_input: the full script. "
            "Use print() to show results; only printed output is returned."
        )

    async def run(self, tool_input: str) -> str:
        result = await self.sandbox.run(tool_input)
        parts = [result.output.rstrip() or "(no output)"]
        if result.output_truncated:
            parts.append(f"[output cut at {self.sandbox.max_output_bytes} bytes]")
        if result.timed_out:
            raise ToolError(
                f"script killed after {self.sandbox.timeout_s:g} s time limit\n" + parts[0]
            )
        if result.output_truncated:
            pass  # we stopped it ourselves; its exit code says nothing about the script
        elif result.exit_code == KILLED:
            parts.append(f"[killed: exit {KILLED}, likely the {self.sandbox.memory} memory limit]")
        elif result.exit_code != 0:
            parts.append(f"[exit code {result.exit_code}]")
        return "\n".join(parts)
