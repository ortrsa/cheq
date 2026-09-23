import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from churn_mcp import data, model, semantic, server
from churn_mcp.config import get_config
from churn_mcp.llm import build as build_llm

app = typer.Typer()


@app.command()
def prepare() -> None:
    config = get_config()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("downloading data, building DuckDB", total=3)
        data.build(config)
        progress.update(task, advance=1, description="training model")
        con = data.connect(config)
        layer = semantic.load(config)
        card = model.build_model_card(con, layer, config)
        progress.update(task, advance=1, description="writing model card")
        model.write_model_card(card, config)
        progress.update(task, advance=1, description="done")
    typer.echo(f"prepared: {config.data.artifacts_dir}")


@app.command()
def serve() -> None:
    config = get_config()
    con = data.connect(config)
    layer = semantic.load(config)
    llm = build_llm(config)
    server.build_server(con, layer, config, llm).run()
