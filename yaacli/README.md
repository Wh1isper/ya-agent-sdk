# YAACLI CLI

TUI reference implementation for [ya-agent-sdk](https://github.com/wh1isper/ya-agent-sdk).

## Usage

With uvx, run:

```bash
uvx yaacli
```

Or to install yaacli globally with uv, run:

```bash
uv tool install yaacli
...
yaacli
```

To update to the latest version:

```bash
uv tool upgrade yaacli
```

Or with pip, run:

```bash
pip install yaacli
...
yaacli
```

Or run as a module:

```bash
python -m yaacli
```

## Development

This package is part of the ya-agent-sdk monorepo. To develop locally:

```bash
cd ya-agent-sdk
uv sync --all-packages
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
