# evals-lite

Local-first LLM eval regression runner. Stdlib only.

## Quickstart

```
evals init                              # scaffold evals.yaml + .evals/
evals run smoke --dry-run               # run the eval set (offline, mock provider)
evals list                              # past runs
evals show smoke                         # per-case results of one run
evals diff smoke smoke2 --fail-on-regression   # CI gate: exit 1 on regression
evals cost smoke                         # total + per-case cost
```

All list/diff/show/cost commands accept `--json` for machine-readable output.

## Providers

`run` calls any OpenAI-compatible `/v1/chat/completions` endpoint:

```
EVALS_API_KEY=sk-... evals run nightly --model gpt-4o-mini
evals run nightly --model gpt-4o --api-key sk-...
```

Default endpoint is `https://api.openai.com`. `--dry-run` uses the built-in
mock provider so CI can run the gate without any key or network access;
pass/fail counts on `--dry-run` are illustrative, not real scores.

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

## CI

```yaml
steps:
  - run: evals run ${{ github.sha }} --dry-run
  - run: evals diff ${{ github.event.before }} ${{ github.sha }} --fail-on-regression
```

The `diff` step fails the build when any case flips pass → fail.

## Limitations

No web UI, no judge-model scoring, sequential runs only.
