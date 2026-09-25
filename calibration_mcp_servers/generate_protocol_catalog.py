"""Generate an MCP-facing catalog of built-in qibocal protocols."""

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "src" / "qibocal" / "protocols"
CATALOG = Path(__file__).with_name("PROTOCOL_CATALOG.md")
CONTROL_SYSTEM_IMPORTS = {"qblox": "Qblox"}


@dataclass
class Parameter:
    name: str
    annotation: str
    description: str
    required: bool


@dataclass
class ParameterClass:
    description: str
    bases: list[str]
    parameters: list[Parameter]


@dataclass
class UpdateCommand:
    name: str
    description: str
    fields: list[str]


def source(node: ast.AST) -> str:
    return ast.unparse(node)


def string_literal(node: ast.AST | None) -> str:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        if isinstance(node.value.value, str):
            value = node.value.value
        else:
            value = ""
    elif isinstance(
        node,
        ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.Module,
    ):
        value = ast.get_docstring(node, clean=True) or ""
    else:
        value = ""
    return " ".join(value.split()).replace("|", "\\|")


def has_default(field: ast.AnnAssign) -> bool:
    return field.value is not None


def parse_parameter_classes(tree: ast.Module) -> dict[str, ParameterClass]:
    classes = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = []
        for index, item in enumerate(node.body):
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue
            description = ""
            if index + 1 < len(node.body):
                next_item = node.body[index + 1]
                if isinstance(next_item, ast.Expr):
                    description = string_literal(next_item)
            fields.append(
                Parameter(
                    name=item.target.id,
                    annotation=source(item.annotation),
                    description=description,
                    required=not has_default(item),
                )
            )
        classes[node.name] = ParameterClass(
            description=string_literal(node),
            bases=[source(base).split(".")[-1] for base in node.bases],
            parameters=fields,
        )
    return classes


def acquisition_details(tree: ast.Module) -> dict[str, tuple[str, str]]:
    details = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "_acquisition" or not node.args.args:
            continue
        annotation = node.args.args[0].annotation
        if annotation is not None:
            details[node.name] = (
                source(annotation).split(".")[-1],
                string_literal(node),
            )
    return details


def call_name(node: ast.Call) -> str:
    """Return a readable dotted name for a call expression."""
    return source(node.func)


def expression_template(node: ast.AST, expressions: dict[str, str]) -> str:
    """Render an expression while expanding local aliases."""
    if isinstance(node, ast.Name) and node.id in expressions:
        return expressions[node.id]
    return source(node)


def field_template(node: ast.AST, expressions: dict[str, str]) -> str:
    """Render a string or f-string used as a platform field path."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + expression_template(value.value, expressions) + "}")
        return "".join(parts)
    return source(node)


def normalize_field(field: str) -> str:
    """Prefix unqualified fields with the qibolab platform parameter path."""
    if field.startswith("platform.calibration"):
        return field.removeprefix("platform.")
    return f"parameters.{field}"


def platform_fields(tree: ast.AST, target_call: ast.Call | None = None) -> list[str]:
    """Extract field paths passed to ``platform.update`` or assigned directly."""
    fields: list[str] = []
    expressions = {
        target.id: source(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and call_name(node) == "platform.update":
            if target_call is not None and node is not target_call:
                continue
            if not node.args or not isinstance(node.args[0], ast.Dict):
                continue
            for key in node.args[0].keys:
                if key is not None:
                    fields.append(field_template(key, expressions))
        elif isinstance(node, ast.Assign):
            fields.extend(platform_assignments(node))
    return list(dict.fromkeys(normalize_field(field) for field in fields))


def platform_assignments(tree: ast.AST) -> list[str]:
    """Extract direct assignments to the calibration object."""
    fields: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            target_source = source(target)
            if target_source.startswith("platform.calibration"):
                fields.append(target_source)
    return list(dict.fromkeys(fields))


def update_commands(
    tree: ast.Module,
    helper_details: dict[str, tuple[str, list[str]]],
    function_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    update_name: str | None = None,
) -> list[UpdateCommand]:
    """Extract calls and direct platform mutations from a protocol update."""
    update_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == (update_name or "_update")
    ]
    if not update_functions and update_name is not None:
        external_update = function_nodes.get(update_name)
        if external_update is not None:
            update_functions = [external_update]
    if not update_functions:
        return []

    commands: list[UpdateCommand] = []
    seen: set[str] = set()

    def called_fields(name: str, visited: set[str] | None = None) -> list[str]:
        visited = visited or set()
        if name in visited:
            return []
        visited.add(name)
        if name in helper_details:
            return helper_details[name][1]
        function = function_nodes.get(name)
        if function is None:
            return []
        fields = platform_fields(function)
        for child in ast.walk(function):
            if isinstance(child, ast.Call):
                fields.extend(called_fields(call_name(child), visited))
        return list(dict.fromkeys(fields))

    for node in ast.walk(update_functions[0]):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        if name in seen:
            continue
        seen.add(name)
        if name.startswith("update."):
            helper = name.removeprefix("update.")
            description, fields = helper_details.get(
                helper,
                (f"Calls `qibocal.update.{helper}`.", []),
            )
        elif name == "platform.update":
            description = "Mutates qibolab platform parameters using dotted paths."
            fields = platform_fields(update_functions[0], node)
        elif name in function_nodes:
            description = string_literal(function_nodes[name]) or (
                "Called while applying the protocol platform update."
            )
            fields = called_fields(name)
            if not fields:
                continue
        elif name.startswith("getattr(update"):
            description = "Calls a dynamically selected qibocal update helper."
            fields = []
        else:
            continue
        commands.append(
            UpdateCommand(name=name, description=description, fields=fields)
        )

    direct_fields = platform_assignments(update_functions[0])
    described_fields = {field for command in commands for field in command.fields}
    direct_fields = [field for field in direct_fields if field not in described_fields]
    if direct_fields:
        commands.append(
            UpdateCommand(
                name="direct platform assignment",
                description="Assigns calibration fields directly on the platform.",
                fields=direct_fields,
            )
        )

    return commands


def update_helper_details() -> dict[str, tuple[str, list[str]]]:
    """Read descriptions and modified fields from qibocal.update helpers."""
    update_path = ROOT / "src" / "qibocal" / "update.py"
    tree = ast.parse(update_path.read_text(encoding="utf-8"))
    details = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            details[node.name] = (
                string_literal(node) or f"Calls `qibocal.update.{node.name}`.",
                platform_fields(node),
            )
    return details


def fallback_description(operation: str) -> str:
    """Describe protocols whose acquisition implementation has no docstring."""
    return f"Runs the `{operation.replace('_', ' ')}` calibration protocol."


def control_system(tree: ast.Module) -> str | None:
    """Identify explicit control-system dependencies from module imports."""
    imported_modules = [
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    ]
    for package, system in CONTROL_SYSTEM_IMPORTS.items():
        if any(package in module.split(".") for module in imported_modules):
            return system
    return None


def protocol_definitions(tree: ast.Module) -> list[tuple[str, str | None]]:
    definitions = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if source(node.value.func) != "Protocol" or not node.value.args:
            continue
        if source(node.value.args[0]) == "_acquisition":
            update = None
            if len(node.value.args) > 3:
                update = source(node.value.args[3])
            for keyword in node.value.keywords:
                if keyword.arg == "update":
                    update = source(keyword.value)
            definitions.append((node.targets[0].id, update))
    return definitions


def inherited_parameters(
    name: str, classes: dict[str, ParameterClass], seen: set[str] | None = None
) -> list[Parameter]:
    seen = seen or set()
    if name in seen or name not in classes:
        return []
    seen.add(name)
    parameter_class = classes[name]
    inherited = [
        parameter
        for base in parameter_class.bases
        for parameter in inherited_parameters(base, classes, seen)
    ]
    return inherited + parameter_class.parameters


def main() -> None:
    classes: dict[str, ParameterClass] = {}
    protocols = []
    parsed_files = []
    helper_details = update_helper_details()
    function_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for path in sorted(PROTOCOLS.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parsed_files.append((path, tree))
        classes.update(parse_parameter_classes(tree))
        function_nodes.update(
            {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        )

    for path, tree in parsed_files:
        acquisitions = acquisition_details(tree)
        for operation, update_name in protocol_definitions(tree):
            details = acquisitions.get("_acquisition")
            if details is None:
                continue
            parameter_type, acquisition_description = details
            parameter_class = classes.get(parameter_type, ParameterClass("", [], []))
            description = (
                acquisition_description
                or parameter_class.description
                or fallback_description(operation)
            )
            protocols.append(
                {
                    "operation": operation,
                    "path": path.relative_to(ROOT),
                    "description": description,
                    "parameters": inherited_parameters(parameter_type, classes),
                    "system": control_system(tree),
                    "update_commands": update_commands(
                        tree, helper_details, function_nodes, update_name
                    ),
                }
            )

    lines = [
        "# Qibocal Protocol Catalog",
        "",
        (
            "Generated by `generate_protocol_catalog.py` from "
            "`src/qibocal/protocols`. Regenerate this file after adding or changing "
            "a protocol."
        ),
        "",
        (
            "All operations also accept `nshots` and `relaxation_time`; qibocal fills "
            "them from the runcard or platform when omitted."
        ),
        "",
        (
            "Use this catalog to map a user's calibration goal to an `operation`, then "
            "ask only for required parameters that cannot be safely inferred. Review the "
            "returned fit/report before calling the approval-gated platform update tool."
        ),
        "",
        "## Platform-agnostic protocols",
        "",
    ]
    generic_protocols = [
        protocol for protocol in protocols if protocol["system"] is None
    ]
    system_protocols = [
        protocol for protocol in protocols if protocol["system"] is not None
    ]
    for protocol in sorted(generic_protocols, key=lambda item: item["operation"]):
        render_protocol(lines, protocol)

    if system_protocols:
        lines.extend(["## System-specific protocols", ""])
        systems = sorted({protocol["system"] for protocol in system_protocols})
        for system in systems:
            lines.extend([f"### {system}", ""])
            for protocol in sorted(
                (item for item in system_protocols if item["system"] == system),
                key=lambda item: item["operation"],
            ):
                render_protocol(lines, protocol)
    CATALOG.write_text("\n".join(lines), encoding="utf-8")


def render_protocol(lines: list[str], protocol: dict) -> None:
    """Append one protocol's description and parameters to the catalog."""
    operation = protocol["operation"]
    path = protocol["path"]
    description = protocol["description"]
    parameters = protocol["parameters"]
    commands = protocol["update_commands"]
    lines.extend([f"### `{operation}`", "", f"Source: `{path}`", ""])
    lines.extend([f"Description: {description}", ""])
    lines.extend(
        ["| Parameter | Type | Required | Description |", "| --- | --- | --- | --- |"]
    )
    for parameter in parameters:
        requirement = "yes" if parameter.required else "no"
        lines.append(
            f"| `{parameter.name}` | `{parameter.annotation}` | {requirement} | "
            f"{parameter.description} |"
        )
    if not parameters:
        lines.append(
            "| None | - | - | This operation has no protocol-specific parameters. |"
        )
    lines.append("")
    lines.extend(["**Platform update fields**", ""])
    if commands:
        lines.extend(["| Field | Description |", "| --- | --- |"])
        for command in commands:
            fields = ", ".join(f"`{field}`" for field in command.fields) or "-"
            lines.append(f"| {fields} | {command.description} |")
    else:
        lines.append("This protocol does not define an `_update` function.")
    lines.append("")


if __name__ == "__main__":
    main()
