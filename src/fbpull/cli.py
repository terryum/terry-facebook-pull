import os
import tempfile
from pathlib import Path

import typer
from dotenv import load_dotenv

from . import classify as classify_mod
from . import cluster as cluster_mod
from . import embed as embed_mod
from . import export as export_mod
from . import filter as filter_mod
from . import parse as parse_mod
from . import report as report_mod
from . import retrieve as retrieve_mod
from . import synthesize as synth_mod
from .paths import ensure_dirs

app = typer.Typer(help="Facebook DYI export → Obsidian pipeline")


def _patch_ssl_certs() -> None:
    """Some corporate setups set SSL_CERT_FILE to a single-cert root that
    a SSL-inspection proxy uses. That root alone can't verify public API
    domains (api.anthropic.com, api.openai.com), so we produce a combined
    bundle (certifi public CAs + that corporate cert) and point
    SSL_CERT_FILE / REQUESTS_CA_BUNDLE at it for this process only.
    No-op when SSL_CERT_FILE already points at a real bundle (>50 KB).
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    if not cert_file:
        return
    p = Path(cert_file)
    if not p.exists() or p.stat().st_size > 50_000:
        return

    import certifi

    combined = Path(tempfile.gettempdir()) / "fbpull-cabundle.pem"
    if not combined.exists() or combined.stat().st_mtime < p.stat().st_mtime:
        public = Path(certifi.where()).read_text()
        corporate = p.read_text()
        combined.write_text(public + "\n" + corporate)
    os.environ["SSL_CERT_FILE"] = str(combined)
    os.environ["REQUESTS_CA_BUNDLE"] = str(combined)


def _bootstrap(no_llm: bool = False) -> None:
    load_dotenv()
    profile_env = os.environ.get("CLAUDE_PROFILE_ENV")
    if profile_env and Path(profile_env).expanduser().exists():
        load_dotenv(Path(profile_env).expanduser(), override=False)
    _patch_ssl_certs()
    if no_llm:
        os.environ["FBPULL_NO_LLM"] = "1"
    ensure_dirs()


@app.command()
def parse() -> None:
    """Stage 1: parse FB JSON → 01_parsed.jsonl"""
    _bootstrap()
    parse_mod.run()


@app.command()
def filter() -> None:
    """Stage 2: heuristic filter → 02_filtered.jsonl"""
    _bootstrap()
    filter_mod.run()


@app.command()
def classify(no_llm: bool = typer.Option(False, "--no-llm")) -> None:
    """Stage 3: Haiku classify → 03_classified.jsonl"""
    _bootstrap(no_llm)
    classify_mod.run(no_llm=no_llm)


@app.command()
def embed(no_llm: bool = typer.Option(False, "--no-llm")) -> None:
    """Stage 4: embed → 04_embeddings.npy + 04_post_ids.json"""
    _bootstrap(no_llm)
    embed_mod.run(no_llm=no_llm)


@app.command()
def cluster() -> None:
    """Stage 5: HDBSCAN + cosine neighbors → 05_clusters.json + 05_neighbors.json"""
    _bootstrap()
    cluster_mod.run()


@app.command()
def synthesize(
    no_llm: bool = typer.Option(False, "--no-llm"),
    include_sensitive: bool = typer.Option(
        False,
        "--include-sensitive",
        help="Synthesize SENSITIVE-flagged categories too. STRICT stays excluded.",
    ),
) -> None:
    """Stage 6: Sonnet synthesize → 06_synthesized.jsonl"""
    _bootstrap(no_llm)
    synth_mod.run(no_llm=no_llm, include_sensitive=include_sensitive)


@app.command()
def export() -> None:
    """Stage 7: write Archive/Synthesized markdown to vault"""
    _bootstrap()
    export_mod.run()


@app.command()
def report() -> None:
    """Generate cluster analysis report (charts + markdown) under vault `_reports/<date>/`."""
    _bootstrap()
    report_mod.run()


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query string to search for relevant posts"),
    top_leaves: int = typer.Option(8, "--top-leaves", help="Stage 1: top-K leaves by centroid"),
    top_posts: int = typer.Option(30, "--top-posts", help="Stage 2: top-N posts overall"),
    tier: str = typer.Option(
        None, "--tier",
        help="Comma-separated importance tiers to keep: core,topic,noise (default = all)",
    ),
    scope: str = typer.Option(
        None, "--scope",
        help="Comma-separated topic_scope filter: personal-family,personal-life,society-politics,society-issues,industry-tech,industry-academic,industry-management",
    ),
) -> None:
    """Stage 8: query → hybrid 2-stage retrieval → markdown + json under `_intermediate/retrieval/`."""
    _bootstrap()
    retrieve_mod.cli_run(
        query, top_leaves=top_leaves, top_posts=top_posts,
        tier=tier, scope=scope,
    )


@app.command(name="open")
def open_note(
    target: str = typer.Argument(
        ...,
        help=(
            "Facebook post id, absolute note path, or vault-relative note path "
            "to open in Obsidian"
        ),
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the Obsidian URI without opening"),
) -> None:
    """Open an archived Facebook note in Obsidian."""
    _bootstrap()
    note = retrieve_mod.open_in_obsidian(target, dry_run=dry_run)
    print(f"[open] {note['obsidian_uri']}")
    print(f"[open] {note['archive_path']}")


@app.command(name="all")
def run_all(
    no_llm: bool = typer.Option(False, "--no-llm", help="Use deterministic stubs (offline/CI)"),
    include_sensitive: bool = typer.Option(
        False,
        "--include-sensitive",
        help="Synthesize SENSITIVE categories too (e.g., 사회비평). STRICT (정치) stays excluded.",
    ),
) -> None:
    """Run all 7 stages in sequence."""
    _bootstrap(no_llm)
    parse_mod.run()
    filter_mod.run()
    classify_mod.run(no_llm=no_llm)
    embed_mod.run(no_llm=no_llm)
    cluster_mod.run()
    synth_mod.run(no_llm=no_llm, include_sensitive=include_sensitive)
    export_mod.run()


if __name__ == "__main__":
    app()
