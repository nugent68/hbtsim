# Releasing hbtsim

1. Bump `version` in `pyproject.toml` (and the changelog line in README if any), commit, push to `main`; wait for the `tests` workflow.
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `release` workflow builds the sdist and wheel, smoke-tests the wheel (`hbtsim catalog validate` from the installed package data) and publishes to PyPI by **trusted publishing**.

One-time setup on pypi.org (project owner): *Your projects → hbtsim → Publishing → add a GitHub publisher* with owner `nugent68`, repository `hbtsim`, workflow `release.yml`, environment `pypi`; create the `pypi` environment in the GitHub repository settings. For a first upload before the project exists on PyPI, use the "pending publisher" form, or upload once by hand: `uv build && uv publish` with a PyPI API token.

Local check before tagging:

```bash
uv build
uv venv /tmp/hbtsim-check && uv pip install -p /tmp/hbtsim-check/bin/python dist/hbtsim-*.whl
/tmp/hbtsim-check/bin/hbtsim catalog validate
/tmp/hbtsim-check/bin/hbtsim snr --target spica --instrument keck_pair --vis2-method analytic --no-newera --time 60
```

The NewEra tables are not in the package: `hbtsim data fetch --target <name>` pulls them from the NERSC portal into the user cache (`hbtsim data path`).
