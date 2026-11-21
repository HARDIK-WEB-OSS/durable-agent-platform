import ast
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dap.agent.tools.base import ToolError

Number = int | float

BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
FUNCTIONS: dict[str, Callable[..., Number]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
CONSTANTS: dict[str, Number] = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_DIGITS = 4000
MAX_INPUT_CHARS = 500


def evaluate(expression: str) -> Number:
    """Evaluate arithmetic only: numbers, + - * / // % **, parentheses, FUNCTIONS, CONSTANTS."""
    if len(expression) > MAX_INPUT_CHARS:
        raise ToolError(f"expression longer than {MAX_INPUT_CHARS} characters")
    try:
        tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid expression: {exc.msg}") from exc
    try:
        return walk(tree.body)
    except ToolError:
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


def walk(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):
            raise ToolError("booleans are not numbers here")
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](walk(node.operand))  # type: ignore[no-any-return]
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponent larger than {MAX_EXPONENT}")
        return check_size(BINARY[type(node.op)](left, right))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return check_size(FUNCTIONS[node.func.id](*(walk(arg) for arg in node.args)))
    raise ToolError(f"unsupported syntax: {ast.unparse(node)!r}")


def check_size(value: Number) -> Number:
    if isinstance(value, int) and abs(value).bit_length() > MAX_DIGITS * 3.33:
        raise ToolError(f"result has more than {MAX_DIGITS} digits")
    return value


def format_number(value: Number) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


@dataclass(frozen=True)
class CalculatorTool:
    name: str = "calculator"
    description: str = (
        "Evaluate one arithmetic expression. tool_input: the expression, e.g. "
        "'(12 / 3) * 2 + sqrt(16)'. Supports + - * / // % ** and abs round min max sqrt "
        "log log10 exp floor ceil sin cos tan pi e."
    )

    async def run(self, tool_input: str) -> str:
        return format_number(evaluate(tool_input))
