"""Live tests of the python_exec Docker sandbox. Skipped when Docker or the image is missing."""

import asyncio
import subprocess

import pytest

from dap.agent.tools.base import ToolError
from dap.agent.tools.python_exec import DockerSandbox, PythonExecTool, SandboxUnavailableError
from dap.config import Settings
from dap.evals.graders import GRADING_SANDBOX_TIMEOUT_S, grade_mbpp

pytestmark = pytest.mark.docker

IMAGE = Settings.model_fields["sandbox_image"].default


def docker_ready() -> bool:
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


if not docker_ready():
    pytest.skip(f"docker or {IMAGE} not available", allow_module_level=True)


@pytest.fixture
def sandbox() -> DockerSandbox:
    return DockerSandbox(IMAGE, timeout_s=20)


async def run(sandbox: DockerSandbox, code: str) -> str:
    result = await sandbox.run(code)
    assert not result.timed_out, result.output
    return result.output


async def test_runs_code_as_nobody_and_returns_output(sandbox: DockerSandbox) -> None:
    out = await run(
        sandbox, "import os, sys\nprint(sys.version_info[:2], os.getuid(), os.getgid())"
    )
    assert out.strip() == "(3, 11) 65534 65534"


async def test_network_access_fails(sandbox: DockerSandbox) -> None:
    code = """
import socket
results = []
for host, port in [("1.1.1.1", 53), ("8.8.8.8", 443)]:
    try:
        socket.create_connection((host, port), timeout=3)
        results.append("CONNECTED")
    except OSError as exc:
        results.append(type(exc).__name__)
try:
    socket.getaddrinfo("example.com", 80)
    results.append("RESOLVED")
except OSError as exc:
    results.append(type(exc).__name__)
print(results)
"""
    out = await run(sandbox, code)
    assert "CONNECTED" not in out and "RESOLVED" not in out, out
    assert out.count("Error") + out.count("gaierror") >= 3, out


async def test_only_loopback_interface_exists(sandbox: DockerSandbox) -> None:
    out = await run(sandbox, "import socket\nprint(sorted(n for _, n in socket.if_nameindex()))")
    assert out.strip() == "['lo']"


@pytest.mark.parametrize("path", ["/etc/evil", "/evil", "/usr/local/lib/evil.py", "/home/evil"])
async def test_writing_outside_tmp_fails(sandbox: DockerSandbox, path: str) -> None:
    code = f"""
try:
    open({path!r}, "w").write("x")
    print("WROTE")
except OSError as exc:
    print(type(exc).__name__, exc.errno)
"""
    out = await run(sandbox, code)
    assert "WROTE" not in out, out
    assert "Error" in out, out


async def test_writing_inside_tmp_works(sandbox: DockerSandbox) -> None:
    code = "open('/tmp/x.txt', 'w').write('hello')\nprint(open('/tmp/x.txt').read())"
    assert (await run(sandbox, code)).strip() == "hello"


async def test_tmp_does_not_survive_between_calls(sandbox: DockerSandbox) -> None:
    await run(sandbox, "open('/tmp/left.txt', 'w').write('x')")
    out = await run(sandbox, "import os\nprint(os.path.exists('/tmp/left.txt'))")
    assert out.strip() == "False"


async def test_no_capabilities_and_no_new_privileges(sandbox: DockerSandbox) -> None:
    code = """
status = open("/proc/self/status").read().splitlines()
print([line for line in status if line.startswith(("CapEff", "NoNewPrivs"))])
"""
    out = await run(sandbox, code)
    assert "CapEff:\\t0000000000000000" in out and "NoNewPrivs:\\t1" in out, out


async def test_timeout_kills_the_container(sandbox: DockerSandbox) -> None:
    quick = DockerSandbox(IMAGE, timeout_s=3)
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await quick.run("print('started', flush=True)\nwhile True: pass")
    assert result.timed_out
    assert loop.time() - started < 15
    code, leftover = await quick.docker_cli("ps", "-aq", "--filter", "name=dap-sbx-")
    assert (code, leftover.strip()) == (0, "")
    with pytest.raises(ToolError, match="time limit"):
        await PythonExecTool(quick).run("while True: pass")


async def test_memory_limit_kills_big_allocation(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("x = bytearray(600 * 1024 * 1024)\nprint('ALLOCATED')")
    assert "ALLOCATED" not in result.output
    assert result.exit_code != 0


async def test_fork_bomb_is_contained(sandbox: DockerSandbox) -> None:
    code = """
import os
n = 0
try:
    for _ in range(500):
        if os.fork() == 0:
            import time; time.sleep(5); os._exit(0)
        n += 1
except OSError as exc:
    print("blocked after", n, type(exc).__name__)
"""
    out = await run(sandbox, code)
    assert "blocked after" in out, out


async def test_runaway_output_is_capped(sandbox: DockerSandbox) -> None:
    small = DockerSandbox(IMAGE, timeout_s=20, max_output_bytes=2000)
    result = await small.run("while True: print('x' * 100)")
    assert result.output_truncated and not result.timed_out
    assert len(result.output) <= 2000


async def test_errors_come_back_as_output_with_exit_code(sandbox: DockerSandbox) -> None:
    out = await PythonExecTool(sandbox).run("raise ValueError('boom')")
    assert "ValueError: boom" in out and "[exit code 1]" in out


async def test_missing_image_is_an_infrastructure_error() -> None:
    absent = DockerSandbox("python:3.11-slim@sha256:" + "f" * 64)
    with pytest.raises(SandboxUnavailableError, match="docker pull"):
        await absent.check_available()


# ---- MBPP grader in the real sandbox --------------------------------------------------------

MBPP_REF = {
    "test_imports": ["import math"],
    "tests": ["assert hyp(3, 4) == 5", "assert hyp(5, 12) == 13"],
}


@pytest.mark.parametrize(
    ("answer", "correct"),
    [
        ("def hyp(a, b):\n    return int(math.sqrt(a * a + b * b))", True),
        ("```python\ndef hyp(a, b):\n    return int(math.hypot(a, b))\n```", True),
        ("def hyp(a, b):\n    return a + b", False),
        ("def hyp(a, b):\n    return 5", False),
        ("import sys\nsys.exit(0)", False),
        ("import os\nprint('ALL_TESTS_PASSED')\nos._exit(0)", False),
        ("def hyp(a, b):\n    while True: pass", False),
        ("this is not python", False),
    ],
)
async def test_mbpp_grader_accepts_only_code_that_passes(
    sandbox: DockerSandbox, answer: str, correct: bool
) -> None:
    grading = DockerSandbox(IMAGE, timeout_s=GRADING_SANDBOX_TIMEOUT_S)
    result = await grade_mbpp(answer, MBPP_REF, grading)
    assert result.correct is correct, result.detail


async def test_python_exec_tool_echoes_final_expression(sandbox: DockerSandbox) -> None:
    tool = PythonExecTool(sandbox)
    assert await tool.run("x = 6 * 7\nx") == "42"
    assert await tool.run("x = 1") == "(no output: use print() to show values)"
