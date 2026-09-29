"""Fast dependency-free .xlsx reader (zip + XML) used for the large KB weekly workbook."""
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
from xml.etree import ElementTree as ET

NS_MAIN = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _col_to_index(cell_ref: str) -> int:
    value = 0
    for char in "".join(ch for ch in cell_ref if ch.isalpha()):
        value = (value * 26) + ord(char.upper()) - 64
    return max(0, value - 1)


def _normalise_sheet_path(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.findall(".//m:t", NS_MAIN)) for si in root.findall("m:si", NS_MAIN)]


def _read_cell(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//m:t", NS_MAIN)).strip()
    value = cell.find("m:v", NS_MAIN)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        idx = int(value.text)
        return shared[idx].strip() if 0 <= idx < len(shared) else value.text.strip()
    return value.text.strip()


def _sheet_paths(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.attrib["Id"]: _normalise_sheet_path(r.attrib["Target"]) for r in rels.findall("r:Relationship", NS_REL)}
    out = []
    for sheet in workbook.findall("m:sheets/m:sheet", NS_MAIN):
        rel_id = sheet.attrib.get(REL_ID)
        if rel_id in rel_map:
            out.append((sheet.attrib.get("name", ""), rel_map[rel_id]))
    return out


def read_workbook_rows(file_bytes: bytes) -> dict[str, list[dict[int, str]]]:
    """Return {sheet_name: [ {col_index: text}, ... ]} (empty cells omitted)."""
    result: dict[str, list[dict[int, str]]] = {}
    with ZipFile(BytesIO(file_bytes)) as zf:
        shared = _load_shared_strings(zf)
        for name, path in _sheet_paths(zf):
            root = ET.fromstring(zf.read(path))
            rows: list[dict[int, str]] = []
            for row in root.findall("m:sheetData/m:row", NS_MAIN):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", NS_MAIN):
                    text = _read_cell(cell, shared)
                    if text:
                        values[_col_to_index(cell.attrib.get("r", "A1"))] = text
                rows.append(values)
            result[name] = rows
    return result
