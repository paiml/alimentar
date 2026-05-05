# This repository has moved

The example tree and book of `alimentar` have been consolidated into the
**APR Cookbook** umbrella project as part of the sovereign-stack
documentation centralization (spec: [docs/specifications/centralize-cookbooks](https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks.md)).

| Where it used to live | Where it lives now |
|-----------------------|--------------------|
| `examples/*.rs` (18 examples) | https://github.com/paiml/apr-cookbook/tree/main/examples/data-loading |
| `book/src/` (mdBook chapters: 100-examples, architecture, backends, cli, dataloader, dataset, datasets, hf-hub, transforms, appendix) | https://github.com/paiml/apr-cookbook/tree/main/book/src/data-loading |

Each migrated example was re-grounded against `contracts/recipe-iiur-v1.yaml`
during the move (per
[iiur-conformance.md](https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks/iiur-conformance.md)
Class 1 retrofit): IIUR doc header with Contract + Citation, RecipeContext
boilerplate, removed allow-blocks, appended `#[cfg(test)] mod tests`.

This repository is now archived (read-only). For new contributions, please use:

- **Cookbook examples and book**: https://github.com/paiml/apr-cookbook
- **Crate source code**: still published on crates.io as `aprender-data`
  (lib name `alimentar`, package name `aprender-data` since v0.31.0).
  See https://crates.io/crates/aprender-data

Last live tag (rollback anchor): `pre-archive-2026-05`

For full migration rationale, see:
https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks.md
