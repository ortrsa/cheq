import typer

from churn_mcp import data, model, semantic, server
from churn_mcp.config import get_config
from churn_mcp.llm import build as build_llm

app = typer.Typer()


@app.command()
def prepare() -> None:
    config = get_config()
    data.build(config)
    con = data.connect(config)
    layer = semantic.load(config)
    model.write_model_card(model.build_model_card(con, layer, config), config)
    typer.echo(f"prepared: {config.data.artifacts_dir}")


@app.command()
def serve() -> None:
    config = get_config()
    con = data.connect(config)
    layer = semantic.load(config)
    llm = build_llm(config)
    server.build_server(con, layer, config, llm).run()
