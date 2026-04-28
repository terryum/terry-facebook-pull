import os

import typer
from dotenv import load_dotenv

from . import classify as classify_mod
from . import cluster as cluster_mod
from . import embed as embed_mod
from . import export as export_mod
from . import filter as filter_mod
from . import parse as parse_mod
from . import synthesize as synth_mod
from .paths import ensure_dirs

app = typer.Typer(help="Facebook DYI export → Obsidian pipeline")


def _bootstrap(no_llm: bool = False) -> None:
    load_dotenv()
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
def synthesize(no_llm: bool = typer.Option(False, "--no-llm")) -> None:
    """Stage 6: Sonnet synthesize → 06_synthesized.jsonl"""
    _bootstrap(no_llm)
    synth_mod.run(no_llm=no_llm)


@app.command()
def export() -> None:
    """Stage 7: write Archive/Synthesized markdown to vault"""
    _bootstrap()
    export_mod.run()


@app.command(name="all")
def run_all(
    no_llm: bool = typer.Option(False, "--no-llm", help="Use deterministic stubs (offline/CI)")
) -> None:
    """Run all 7 stages in sequence."""
    _bootstrap(no_llm)
    parse_mod.run()
    filter_mod.run()
    classify_mod.run(no_llm=no_llm)
    embed_mod.run(no_llm=no_llm)
    cluster_mod.run()
    synth_mod.run(no_llm=no_llm)
    export_mod.run()


if __name__ == "__main__":
    app()
