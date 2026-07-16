# Security policy

## Supported version

Security fixes target the latest tagged Cortex release.

## Reporting

Please report suspected vulnerabilities privately to the repository owner through GitHub's private vulnerability reporting feature when enabled. Do not include real memory content, API keys, dashboard passwords, private hostnames, or vault documents in a public issue.

Include a minimal redacted reproduction, affected version, impact, and suggested mitigation if known.

## Important deployment boundary

The built-in dashboard server is designed to bind to localhost. A public deployment must use a maintained TLS reverse proxy and authentication. Cortex is not a general-purpose identity provider or internet-facing application server.

Review Copilot is disabled unless the dashboard receives an explicit enable flag, HTTPS-or-loopback endpoint, model, and provider credential. Each request crosses the local trust boundary with two bounded memory excerpts, proposal metadata, and the operator's copilot conversation; the dashboard discloses the provider and model before use. Provider output is untrusted, schema-validated, recommendation-only, and cannot bypass the authenticated review confirmation. Copilot dialogue is stored in `review_copilot_interpretations` for audit and never enters recallable memory.
