# RPT Project Governance

RPT is published as an open research and engineering project under the Apache
License 2.0.

## Public Repositories

- [`RPT`](https://github.com/cesarfavero/RPT): research source of truth, notebooks, result artifacts, papers, validation notes, and reproducibility docs.
- [`rpt-runtime`](https://github.com/cesarfavero/rpt-runtime): runtime and CLI layer for packaged RPT artifacts.
- [`rpt-model-bundles`](https://github.com/cesarfavero/rpt-model-bundles): bundle manifests, checksums, release metadata, and bundle distribution notes.

The website repository is operational infrastructure and is not part of the
public replication surface.

## Issues

Use GitHub Issues in the repository closest to the problem:

- `RPT`: methodology, notebooks, results, documentation, reproducibility.
- `rpt-runtime`: CLI, runtime loading, environment support, execution errors.
- `rpt-model-bundles`: manifests, checksums, bundle metadata, release notes.

Good issue reports include:

- environment details;
- exact command or notebook step;
- expected behavior;
- observed behavior;
- logs or result JSON when available.

## Licensing Boundaries

RPT source code, documentation, notebooks, and repository materials are Apache
2.0 unless a file states otherwise.

Third-party model weights, datasets, checkpoints, and trademarks are not
relicensed by RPT. Gemma-family assets remain subject to the applicable Gemma
Terms of Use.

## Replication Policy

The project should remain reproducible from public materials whenever the
underlying third-party licenses allow it.

The expected public replication path is:

1. read `README.md`;
2. read `gemma4_theory/README.md`;
3. follow `REPRODUCIBILITY.md`;
4. inspect result artifacts under `gemma4_theory/results/`;
5. use `rpt-runtime` for runtime and CLI work;
6. use `rpt-model-bundles` for bundle metadata and release references.
