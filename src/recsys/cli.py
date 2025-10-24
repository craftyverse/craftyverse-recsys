import typer

app = typer.Typer(help="Minimal Recsys CLI")

# Make 'name' a positional argument
@app.command()
def hello(name: str = typer.Argument("world", help="Name to greet")):
    typer.echo(f"Hello {name}!")

if __name__ == "__main__":
    app()