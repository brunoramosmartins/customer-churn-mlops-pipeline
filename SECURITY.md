# Security

## Supported versions

Security fixes are applied to the **default branch** (`main`). Release tags (e.g. `v1.0.0`) reflect portfolio snapshots; use `main` for the latest fixes.

## Reporting a vulnerability

If you believe you found a security issue (for example, unsafe deserialization or secret exposure in the repo), please **do not** open a public issue with exploit details.

Instead, contact the repository owner via **GitHub private vulnerability reporting** (if enabled on the repo) or a **private** message channel they publish on their profile / portfolio site.

This is a small open-source portfolio project; response times are best-effort.

## Scope notes

- **Do not commit** real credentials, API keys, or full production datasets. Use `.env` locally (gitignored); see `.env.example`.
- The training and serving stack uses **joblib** and **YAML** from trusted local paths; only load artifacts from sources you trust.
