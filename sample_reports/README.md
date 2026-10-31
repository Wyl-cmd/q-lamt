# Result snapshots

These are small `report.json` / `manifest.json` snapshots from the experiments
described in `RESULTS.md`. They are **not** loadable models: no ledgers,
undo snapshots, or release weights are included.

- `0.5b-*`: single-call, multi-round, 12-item EOS/no-EOS summaries;
- `7b-*`: Valdoria, identity, release manifests;
- `7b-version-manifests/`: the 7B version chain.

Historical note: the identity target `NGT Labs` is a **synthetic test entity**
used in the experiments, not the project name. The package/repository was
renamed **Q-LAMT** after these runs; old manifests may still say
`ngt-0.1.0` or contain the former package name in `code_version`.
