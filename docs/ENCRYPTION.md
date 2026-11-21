# Encryption at rest

Status: **the capability is not present on this machine yet.** `scripts/encryption_readiness.py`
reports it and refuses to pretend. This document is the design, the key story and the migration
procedure, so that the day the dependency exists the switch is a command rather than a project.

## What "at rest" has to mean

The database is a single SQLite file holding every recalled fact, including anything the agent
learned about people, work and money. At rest means: a copy of that file, taken without the key,
does not yield plaintext. It does **not** protect against an attacker who is already root on the
host with the key mounted — that is a different problem (see Threat model).

## What we need

1. **An encryption extension.** SQLCipher is the realistic option: it is SQLite with AES-256 page
   encryption, and it is a drop-in at the connection level (`PRAGMA key = ...` before any other
   statement). Vanilla SQLite has no encryption: `PRAGMA cipher_version` returns NULL, and
   `PRAGMA key` is silently accepted and ignored — which is exactly the trap this repo must avoid.
2. **Key material**, 32+ random bytes, mode `0600`, outside the repository and outside any backup
   of the database.
3. **A verified migration**, because a half-migrated store is worse than a plaintext one.

## Key story

- Generated once: `openssl rand -base64 48 > /etc/cortex/store.key && chmod 600 /etc/cortex/store.key`.
- Read at startup by the service, never written to logs, never echoed by tooling. The readiness
  script checks the file's mode and size and reports neither its contents nor its length beyond
  a byte count.
- Rotation: decrypt-to-temp with the old key, re-encrypt with the new one, verify row counts and a
  content hash, swap, then destroy the temp. Rotation is a drill, not a hope: `backup_restore_drill`
  already establishes the verify-then-swap shape.
- The key is **not** in the database backup. Losing the key loses the memory; that is the point,
  and it must be stated wherever a backup routine is documented.

## Migration

```
scripts/encryption_readiness.py --db ~/.hermes/cortex/cortex.db   # must exit 0
scripts/backup_restore_drill.py ~/.hermes/cortex/cortex.db        # prove a restore first
# then, with SQLCipher available:
sqlcipher ~/.hermes/cortex/cortex.db "ATTACH DATABASE 'cortex.enc' AS enc KEY 'x'; SELECT sqlcipher_export('enc');"
# verify: same row counts, same sample hash, integrity_check on the encrypted copy
# swap: stop service, move plaintext aside, move encrypted into place, start, run the frozen set
```

The frozen set is the acceptance test: `hit@6 / recall@6 / MRR` must be identical after the swap,
because encryption changes how pages are stored, not what retrieval computes.

## Threat model — what this does and does not buy

| Attacker | Covered |
|---|---|
| Someone with a copy of the file (backup leak, disk image, stolen laptop) | yes |
| A process running as the same user, while the service is up | no |
| Root on the host, key mounted | no |
| A backup that includes the key file | **no — this is the real risk, and the reason the key lives outside** |

If the threat is "the disk leaves the building", full-disk encryption (LUKS) covers it and needs
no application change. SQLCipher matters when the file is copied around — backups, sync, a
dashboard export — and those copies must stay useless without the key.

## Decision required

The dependency is a system change, so it needs a decision rather than a commit:

1. **Install SQLCipher** (apt `libsqlite3-mod-sqlcipher` plus a Python binding, or build it) and
   migrate the live store. Highest protection for file copies; touches the live service.
2. **Rely on host-level disk encryption** (LUKS on the volume holding `~/.hermes/cortex`). No code
   change, covers disk theft, does nothing for a copied file.
3. **App-level field encryption** of a defined sensitive subset (finance, health, credentials
   pointers). Works with vanilla SQLite, but per-field crypto in a query path is a large change
   and slows recall.

Until one is chosen, the readiness script exits 1 and that is the honest state.
