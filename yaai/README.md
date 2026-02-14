# YAAI CLI

TUI reference implementation for [ya-agent-sdk](https://github.com/wh1isper/ya-agent-sdk).

## Usage

With uvx, run:

```bash
uvx yaai
```

Or to install yaai globally with uv, run:

```bash
uv tool install yaai
...
yaai
```

To update to the latest version:

```bash
uv tool upgrade yaai
```

Or with pip, run:

```bash
pip install yaai
...
yaai
```

Or run as a module:

```bash
python -m yaai
```

## Development

This package is part of the ya-agent-sdk monorepo. To develop locally:

```bash
cd ya-agent-sdk
uv sync --all-packages
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
