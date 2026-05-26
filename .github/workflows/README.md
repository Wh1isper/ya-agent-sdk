# GitHub Actions trigger policy

This directory keeps workflow path filters narrow so pull requests run the checks that match the files they touch.

## Baseline checks

`main.yml` is the repository-wide safety net. It runs on every pull request and main-branch push, and concurrency cancels older runs for the same pull request.

## Focused workflows

| workflow                      | trigger scope                                                                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file-search-benchmarks.yml`  | File search benchmark harness, SDK filesystem tool code/tests, FileOperator surface, and `ya-ripgrep-core` source/package metadata.                |
| `cli-tests.yml`               | YAACLI source/tests, CLI-facing SDK APIs, bundled skills, and skill sync/build scripts.                                                            |
| `shell-sandbox.yml`           | SDK shell sandbox/process/local environment code and Claw shell sandbox adapter/tests.                                                             |
| `claw-image.yaml`             | Docker packaging inputs for the Claw image: Dockerfile, web app bundle inputs, runtime source/package metadata, lock files, and package manifests. |
| `platform-image.yaml`         | Docker packaging inputs for the WIP platform image.                                                                                                |
| `ya-claw-workspace-image.yml` | Workspace image Dockerfile and bundled workspace skill/image inputs.                                                                               |

## Guidelines

- Prefer package source paths over whole package globs when a workflow tests one feature area.
- Keep README/spec-only changes out of heavyweight image workflows.
- Keep `uv.lock` in image workflow triggers because Docker builds use frozen dependency resolution.
- Add `concurrency` to PR workflows so new pushes cancel older runs for the same pull request.
- Let `main.yml` provide broad correctness coverage; focused workflows should cover specialized risk.
