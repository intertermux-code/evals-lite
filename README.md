# evals-lite

[![PyPI](https://img.shields.io/pypi/v/evals-lite)](https://pypi.org/project/evals-lite/)
[![Tests](https://github.com/intertermux-code/evals-lite/actions/workflows/test.yml/badge.svg)](https://github.com/intertermux-code/evals-lite/actions/workflows/test.yml)

Local-first LLM eval regression runner. Stdlib only.

## Install

```bash
pip install evals-lite
evals --help
```

From a source checkout, `python -m evals --help` works without installing (or
`pip install -e .` for the `evals` command).

## Quickstart

```
evals init                              # scaffold evals.yaml + .evals/
evals run smoke --dry-run               # run the eval set (offline, mock provider)
evals list                              # past runs
evals show smoke                         # per-case results of one run
evals diff smoke smoke2 --fail-on-regression   # CI gate: exit 1 on regression
evals cost smoke                         # total + per-case cost
evals cost --by model                    # spend rolled up per model
evals report smoke --out reports/smoke.md  # Markdown report for a run
```

All list/diff/show/cost commands accept `--json` for machine-readable output;
`report` uses `--format json`.

## Providers

Every provider is reached through an OpenAI-compatible `/v1/chat/completions`
endpoint, so a preset is just three values: base URL, API-key env var and
default model.

```
evals run nightly --provider openai                  # https://api.openai.com
evals run nightly --provider anthropic               # https://api.anthropic.com
evals run nightly --provider ollama                  # http://localhost:11434
```

| preset | key env var | default model |
| --- | --- | --- |
| `openai` (default) | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| `ollama` | none | `llama3.1` |

`--provider` defaults to `openai`, so an unqualified run behaves exactly as
before. Any flag beats the preset, which is how you point a name at a gateway
or pick a different model:

```
OPENAI_API_KEY=sk-... evals run nightly --model gpt-4o-mini
evals run nightly --provider anthropic --base-url https://gateway.internal
evals run nightly --base-url http://localhost:11434/v1   # any OpenAI-compatible server
evals run nightly --model gpt-4o --api-key sk-...        # last resort: the key is in your shell history
```

Key lookup order is `--api-key`, then the preset's env var, then
`EVALS_API_KEY`. An unknown `--provider` exits 1 and lists the known names.
Providers that do not speak the OpenAI-compatible path (or keys you would
rather not export) go through a local gateway, addressed with `--base-url`.

`--dry-run` uses the built-in mock provider so CI can run the gate without any
key or network access; pass/fail counts on `--dry-run` are illustrative, not
real scores.

Cases run one after another. `--concurrency` is refused above 1.

## Eval-set format

`evals.yaml`:

```yaml
name: smoke
cases:
  - id: greeting
    input: "Say hello in one word."
    expected: "hello"
    scorer: contains
```

Fields: id, input, expected (optional), scorer or scorers (list). Case passes if ALL scorers pass.

Scorers: exact, contains, fuzzy (threshold default 0.6).

## Cost

Built-in price table; unknown models => NULL.

`evals cost [ref]` reports one run (defaults to the latest), as before.
`--by` rolls up across every run in the ledger:

```
evals cost --by model                # runs, cases, tokens and spend per model
evals cost --by run                  # spend per run, oldest first
evals cost --by model --json         # same numbers, machine-readable
```

```
cost by model
model                runs cases  prompt completion     cost_usd
gpt-4o                  4     8   12000       6400     0.156000
gpt-4o-mini             9    18    9000       4100     0.003810
llama3.1                3     6    3200       1800            -
```

Models with no price entry show `-` because their cost is unknown, not zero.
`--by` cannot take a ref — it aggregates the whole ledger.

## Reports

`evals report [ref]` renders one run (defaults to the latest) as Markdown: run
metadata, a per-case pass/fail table with the scorer detail and a snippet of
each output, a per-scorer pass-rate breakdown, and a cost summary.

```
evals report smoke                       # to stdout
evals report smoke --out reports/sm.md   # to a file
evals report smoke --format json         # same data as JSON
```

Reports are safe to paste into a PR or an issue: model output is flattened
into one table cell, `|` is escaped, and long outputs are truncated.

## CI

```yaml
steps:
  - run: evals run ${{ github.sha }} --dry-run
  - run: evals diff ${{ github.event.before }} ${{ github.sha }} --fail-on-regression
  - run: evals report ${{ github.sha }} --out reports/${{ github.sha }}.md
```

The `diff` step fails the build when any case flips pass → fail. The report step is
optional and pairs well with `actions/upload-artifact`.

## Releasing

Releases are published to PyPI by GitHub Actions using OIDC trusted publishing,
so no API token is stored anywhere:

1. Bump the version in `pyproject.toml` and `evals/__init__.py`. A test keeps
   the two in sync.
2. Tag the release and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   The publish workflow checks the tag against `pyproject.toml` and fails if
   they differ; otherwise it builds an sdist and a wheel, runs `twine check` on
   both and uploads them to PyPI. Running the workflow manually does the same
   thing without a tag.
3. Optionally create the matching GitHub release for the tag.

Before the first release, a maintainer has to register the `evals-lite` project
on PyPI and configure a trusted publisher for it: owner/repository
`intertermux-code/evals-lite`, workflow name `Publish` (filename
`.github/workflows/publish.yml`), and no environment (the workflow does not
declare one).

## Testing

```bash
pip install -e . pytest
python -m pytest
```

The suite runs on Python 3.11, 3.12 and 3.13 in CI. Building a release
locally needs `python -m build` and `twine check dist/*`; the wheel can then be
installed into a fresh venv and smoke-tested with `evals --help`.

## Limitations

No web UI, no judge-model scoring, sequential runs only. Presets only cover
providers reachable through an OpenAI-compatible chat endpoint; anything else
needs a gateway in front of it.
