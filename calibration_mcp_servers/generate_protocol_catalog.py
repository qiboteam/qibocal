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


def protocol_names(tree: ast.Module) -> list[str]:
    names = []
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
            names.append(node.targets[0].id)
    return names


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
    for path in sorted(PROTOCOLS.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parsed_files.append((path, tree))
        classes.update(parse_parameter_classes(tree))

    for path, tree in parsed_files:
        acquisitions = acquisition_details(tree)
        for operation in protocol_names(tree):
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


if __name__ == "__main__":
    main()
