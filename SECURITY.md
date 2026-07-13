# Security policy

## Supported version

Security fixes target the latest tagged Cortex release.

## Reporting

Please report suspected vulnerabilities privately to the repository owner through GitHub's private vulnerability reporting feature when enabled. Do not include real memory content, API keys, dashboard passwords, private hostnames, or vault documents in a public issue.

Include a minimal redacted reproduction, affected version, impact, and suggested mitigation if known.

## Important deployment boundary

The built-in dashboard server is designed to bind to localhost. A public deployment must use a maintained TLS reverse proxy and authentication. Cortex is not a general-purpose identity provider or internet-facing application server.
