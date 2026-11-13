import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_CSV = BASE_DIR / "data" / "metadata" / "vbpl_master.csv"
RAW_TEXT_DIR = BASE_DIR / "data" / "raw_text"
DEFAULT_OUTPUT = BASE_DIR / "data" / "legal_reference_graph.html"


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def load_graph_data() -> Dict[str, List[dict]]:
    with METADATA_CSV.open(encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        rows = list(reader)

    documents = []
    documents_by_number = {}
    text_paths = {path.stem.casefold(): path for path in RAW_TEXT_DIR.glob("*.txt")}

    for row in rows:
        number = (row.get("Số ký hiệu") or "").strip()
        if not number:
            continue
        document_id = (row.get("ID") or number).strip()
        documents.append(
            {
                "id": document_id,
                "number": number,
                "title": (row.get("Tên văn bản") or "").strip(),
                "type": (row.get("Loại văn bản") or "").strip(),
                "issuer": (row.get("Cơ quan ban hành") or "").strip(),
                "validity": (row.get("Tình trạng hiệu lực") or "").strip(),
                "text_available": sanitize_filename(number).casefold() in text_paths,
            }
        )
        documents_by_number[number.casefold()] = document_id

    patterns = [
        (re.escape(document["number"]), document["number"].casefold())
        for document in sorted(documents, key=lambda item: len(item["number"]), reverse=True)
    ]
    reference_pattern = re.compile(
        r"(?<![\w/-])(" + "|".join(pattern for pattern, _ in patterns) + r")(?![\w/-])",
        re.IGNORECASE,
    )
    edges = Counter()

    for document in documents:
        source_id = document["id"]
        text_path = text_paths.get(sanitize_filename(document["number"]).casefold())
        if text_path is None:
            continue
        content = text_path.read_text(encoding="utf-8", errors="replace")
        for match in reference_pattern.finditer(content):
            target_id = documents_by_number.get(match.group(0).casefold())
            if target_id and target_id != source_id:
                edges[(source_id, target_id)] += 1

    return {
        "nodes": documents,
        "edges": [
            {"source": source, "target": target, "mentions": mentions}
            for (source, target), mentions in edges.items()
        ],
    }


def render_html(graph: Dict[str, List[dict]], graph_kind: str = "lightrag") -> str:
    graph_json = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    if graph_kind == "references":
        config = {
            "node_label": "document",
            "edge_label": "reference relationship",
            "title": "Legal Document Reference Graph",
            "search": "Search document number or title",
            "aria": "Reference graph between legal documents",
            "intro": "Each node is a document in the catalog. Edge A → B indicates document A references document B's number.",
            "type_label": "Document type",
            "source_label": "Issuing authority",
            "outgoing": "References to",
            "incoming": "Referenced by",
            "legend": "Blue nodes have document type info; orange nodes are unclassified.",
            "show_weight": True,
        }
    elif graph_kind == "lightrag":
        config = {
            "node_label": "entity",
            "edge_label": "LightRAG relationship",
            "title": "LightRAG Knowledge Graph · Legal Documents",
            "search": "Search entity or description",
            "aria": "LightRAG Knowledge Graph",
            "intro": "Each node is an entity extracted by LightRAG from legal documents. Edge represents semantic relationship identified by LightRAG.",
            "type_label": "Entity type",
            "source_label": "Citation source",
            "outgoing": "Outgoing relationships",
            "incoming": "Incoming relationships",
            "legend": "Blue nodes have entity type; orange nodes are unclassified; gray nodes have no description.",
            "show_weight": False,
        }
    else:
        raise ValueError(f"Unknown graph kind: {graph_kind}")
    config_json = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c")
    intro = (
        f"<h2>Explore graph</h2><p>{config['intro']}</p>"
        "<p>Drag nodes to arrange, drag background to pan, scroll to zoom in/out. "
        "Select node to view details and related relationships.</p>"
        f"<div class=\"legend\">{config['legend']}</div>"
    )
    return (
        PAGE.replace("__GRAPH_DATA__", graph_json)
        .replace("__GRAPH_CONFIG__", config_json)
        .replace("__GRAPH_TITLE__", config["title"])
        .replace("__GRAPH_SEARCH__", config["search"])
        .replace("__GRAPH_ARIA__", config["aria"])
        .replace("__GRAPH_INTRO__", intro)
    )


def generate_graph(output_path: Optional[Path] = None) -> Path:
    output = output_path or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(load_graph_data(), "references"), encoding="utf-8")
    return output


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__GRAPH_TITLE__</title>
  <style>
    :root { color-scheme: dark; font-family: "Segoe UI", sans-serif; background: #10151e; color: #e7edf5; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: flex; flex-direction: column; }
    header { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 14px 18px; border-bottom: 1px solid #293445; }
    h1 { margin: 0 auto 0 0; font-size: 19px; font-weight: 650; }
    .stats, .hint { color: #a7b3c4; font-size: 13px; }
    input, button { color: inherit; background: #1a2330; border: 1px solid #36465d; border-radius: 6px; padding: 8px 10px; }
    input { width: min(320px, 70vw); }
    button { cursor: pointer; }
    button:hover { background: #263448; }
    main { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 300px; }
    #graph { width: 100%; height: 100%; min-height: 520px; display: block; cursor: grab; }
    #graph:active { cursor: grabbing; }
    aside { padding: 18px; border-left: 1px solid #293445; overflow: auto; }
    aside h2 { font-size: 16px; margin: 0 0 12px; }
    aside p { color: #c0cad8; line-height: 1.5; overflow-wrap: anywhere; }
    .legend { margin-top: 20px; color: #a7b3c4; font-size: 13px; line-height: 1.8; }
    .dot { display: inline-block; width: 9px; height: 9px; margin-right: 7px; border-radius: 50%; }
    a { color: #7ec7ff; }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; grid-template-rows: minmax(430px, 65vh) auto; }
      aside { border-left: 0; border-top: 1px solid #293445; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__GRAPH_TITLE__</h1>
    <input id="search" type="search" placeholder="__GRAPH_SEARCH__">
    <button id="fit">Fit to view</button>
    <span class="stats" id="stats"></span>
  </header>
  <main>
    <canvas id="graph" aria-label="__GRAPH_ARIA__"></canvas>
    <aside id="details">__GRAPH_INTRO__</aside>
  </main>
  <script>
    const DATA = __GRAPH_DATA__;
    const CONFIG = __GRAPH_CONFIG__;
    const canvas = document.getElementById("graph");
    const ctx = canvas.getContext("2d");
    const search = document.getElementById("search");
    const details = document.getElementById("details");
    const nodeById = new Map(DATA.nodes.map(node => [node.id, node]));
    const states = DATA.nodes.map((node, i) => {
      const angle = i * 2.399963;
      const radius = 30 + Math.sqrt(i + 1) * 13;
      return { node, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, vx: 0, vy: 0 };
    });
    const stateById = new Map(states.map(state => [state.node.id, state]));
    const edges = DATA.edges.map(edge => ({ ...edge, a: stateById.get(edge.source), b: stateById.get(edge.target) }));
    let width = 0, height = 0, scale = 1, offsetX = 0, offsetY = 0;
    let selected = null, hovered = null, dragged = null, panning = false, pointer = null;
    let query = "";

    document.getElementById("stats").textContent =
      `${DATA.nodes.length} ${CONFIG.node_label} · ${DATA.edges.length} ${CONFIG.edge_label}`;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      width = rect.width; height = rect.height;
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function screen(state) {
      return { x: width / 2 + offsetX + state.x * scale, y: height / 2 + offsetY + state.y * scale };
    }
    function matches(state) {
      return !query || `${state.node.number} ${state.node.title} ${state.node.type}`.toLocaleLowerCase("vi").includes(query);
    }
    function color(state) {
      if (!state.node.title) return "#65748b";
      return state.node.type ? "#4adea1" : "#f4b860";
    }
    function draw() {
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.lineWidth = 1 / scale;
      for (const edge of edges) {
        if (!edge.a || !edge.b) continue;
        const visible = matches(edge.a) || matches(edge.b);
        ctx.strokeStyle = visible ? "rgba(141, 164, 193, .25)" : "rgba(141, 164, 193, .035)";
        ctx.lineWidth = (1 + Math.min(edge.mentions, 5) * .35) / scale;
        const angle = Math.atan2(edge.b.y - edge.a.y, edge.b.x - edge.a.x);
        const endX = edge.b.x - Math.cos(angle) * 5;
        const endY = edge.b.y - Math.sin(angle) * 5;
        ctx.beginPath(); ctx.moveTo(edge.a.x, edge.a.y); ctx.lineTo(endX, endY); ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - Math.cos(angle - .45) * 7 / scale, endY - Math.sin(angle - .45) * 7 / scale);
        ctx.lineTo(endX - Math.cos(angle + .45) * 7 / scale, endY - Math.sin(angle + .45) * 7 / scale);
        ctx.closePath(); ctx.fill();
      }
      for (const state of states) {
        const isMatch = matches(state), active = selected === state || hovered === state;
        ctx.globalAlpha = isMatch ? 1 : .12;
        ctx.fillStyle = color(state);
        ctx.beginPath(); ctx.arc(state.x, state.y, (active ? 7 : 4.5) / scale, 0, Math.PI * 2); ctx.fill();
        if (active || (query && isMatch)) {
          ctx.font = `${12 / scale}px Segoe UI`;
          ctx.fillStyle = "#e7edf5";
          ctx.fillText(state.node.number, state.x + 8 / scale, state.y - 7 / scale);
        }
      }
      ctx.restore();
      requestAnimationFrame(draw);
    }
    function worldPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return { x: (event.clientX - rect.left - width / 2 - offsetX) / scale,
        y: (event.clientY - rect.top - height / 2 - offsetY) / scale };
    }
    function nearest(point) {
      let found = null, best = 13 / scale;
      for (const state of states) {
        const distance = Math.hypot(state.x - point.x, state.y - point.y);
        if (distance < best) { found = state; best = distance; }
      }
      return found;
    }
    function select(state) {
      selected = state;
      if (!state) return;
      const node = state.node;
      const related = edges.filter(edge => edge.source === node.id || edge.target === node.id);
      const links = related.map(edge => {
        const outgoing = edge.source === node.id;
        const other = nodeById.get(outgoing ? edge.target : edge.source);
        const relation = edge.type ? ` — ${escapeHtml(edge.type)}` : "";
        const description = edge.description ? `<br>${escapeHtml(edge.description)}` : "";
        const weight = CONFIG.show_weight ? ` (${edge.mentions} lần)` : "";
        return `<li>${outgoing ? CONFIG.outgoing : CONFIG.incoming} <b>${escapeHtml(other.number)}</b>${relation}${weight}${description}</li>`;
      }).join("");
      details.innerHTML = `<h2>${escapeHtml(node.number)}</h2>
        <p>${escapeHtml(node.title || "Chưa có mô tả")}</p>
        <p>${CONFIG.type_label}: ${escapeHtml(node.type || "Chưa phân loại")}<br>
        ${CONFIG.source_label}: ${escapeHtml(node.issuer || "Chưa rõ")}</p>
        <p>${related.length} ${CONFIG.edge_label}</p><ul>${links || "<li>Chưa tìm thấy quan hệ liên kết.</li>"}</ul>
        <div class="legend"><a href="#" id="reset-details">Bỏ chọn</a></div>`;
      document.getElementById("reset-details").addEventListener("click", event => {
        event.preventDefault(); selected = null; showIntro();
      });
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[char]);
    }
    function showIntro() {
      details.innerHTML = `<h2>Khám phá đồ thị</h2>
        <p>${escapeHtml(CONFIG.intro)}</p>
        <p>Kéo nút để sắp xếp, kéo nền để di chuyển, cuộn để phóng to/thu nhỏ. Chọn nút để xem chi tiết và các quan hệ liên quan.</p>
        <div class="legend">${escapeHtml(CONFIG.legend)}</div>`;
    }
    function fit() {
      if (!states.length) return;
      const xs = states.map(state => state.x), ys = states.map(state => state.y);
      const spanX = Math.max(...xs) - Math.min(...xs) + 60;
      const spanY = Math.max(...ys) - Math.min(...ys) + 60;
      scale = Math.min(width / spanX, height / spanY, 2);
      offsetX = offsetY = 0;
    }
    canvas.addEventListener("pointerdown", event => {
      canvas.setPointerCapture(event.pointerId);
      pointer = { x: event.clientX, y: event.clientY };
      const point = worldPoint(event);
      dragged = nearest(point); panning = !dragged;
    });
    canvas.addEventListener("pointermove", event => {
      const point = worldPoint(event);
      if (dragged) { dragged.x = point.x; dragged.y = point.y; dragged.vx = dragged.vy = 0; }
      else if (panning && pointer) {
        offsetX += event.clientX - pointer.x; offsetY += event.clientY - pointer.y;
      } else hovered = nearest(point);
      pointer = { x: event.clientX, y: event.clientY };
    });
    canvas.addEventListener("pointerup", () => { dragged = null; panning = false; pointer = null; });
    canvas.addEventListener("pointercancel", () => { dragged = null; panning = false; pointer = null; });
    canvas.addEventListener("click", event => {
      const point = worldPoint(event);
      select(nearest(point));
    });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const before = worldPoint(event);
      scale = Math.max(.25, Math.min(4, scale * (event.deltaY < 0 ? 1.1 : .9)));
      const after = worldPoint(event);
      offsetX += (after.x - before.x) * scale;
      offsetY += (after.y - before.y) * scale;
    }, { passive: false });
    search.addEventListener("input", () => { query = search.value.trim().toLocaleLowerCase(); });
    document.getElementById("fit").addEventListener("click", fit);

    function simulate() {
      const cellSize = 120;
      const cells = new Map();
      const forces = new Map();
      for (const state of states) {
        forces.set(state, { x: -state.x * .0008, y: -state.y * .0008 });
        const key = `${Math.floor(state.x / cellSize)},${Math.floor(state.y / cellSize)}`;
        if (!cells.has(key)) cells.set(key, []);
        cells.get(key).push(state);
      }
      for (const state of states) {
        if (state === dragged) continue;
        const cellX = Math.floor(state.x / cellSize), cellY = Math.floor(state.y / cellSize);
        const force = forces.get(state);
        for (let x = cellX - 1; x <= cellX + 1; x++) {
          for (let y = cellY - 1; y <= cellY + 1; y++) {
            for (const other of cells.get(`${x},${y}`) || []) {
              if (other === state) continue;
              const dx = state.x - other.x, dy = state.y - other.y;
              const distance2 = dx * dx + dy * dy + 1;
              const repulsion = 900 / distance2;
              force.x += dx * repulsion; force.y += dy * repulsion;
            }
          }
        }
      }
      for (const edge of edges) {
        if (!edge.a || !edge.b) continue;
        const dx = edge.b.x - edge.a.x, dy = edge.b.y - edge.a.y;
        const distance = Math.hypot(dx, dy) || 1;
        const attraction = (distance - 105) * .002;
        const forceA = forces.get(edge.a), forceB = forces.get(edge.b);
        forceA.x += dx / distance * attraction; forceA.y += dy / distance * attraction;
        forceB.x -= dx / distance * attraction; forceB.y -= dy / distance * attraction;
      }
      for (const state of states) {
        if (state === dragged) continue;
        const force = forces.get(state);
        state.vx = (state.vx + force.x) * .84; state.vy = (state.vy + force.y) * .84;
        state.x += state.vx; state.y += state.vy;
      }
    }
    setInterval(simulate, 30);
    new ResizeObserver(() => { resize(); fit(); }).observe(canvas);
    resize(); fit(); draw();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create interactive graph from legal document corpus.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML file path.")
    args = parser.parse_args()
    graph = load_graph_data()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(graph), encoding="utf-8")
    print(
        f"Created {output} with {len(graph['nodes'])} documents "
        f"and {len(graph['edges'])} reference relationships."
    )


if __name__ == "__main__":
    main()
