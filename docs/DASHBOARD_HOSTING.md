# Host your own Cortex Brain dashboard

This guide uses `brain.example.com` as a placeholder. Replace it with a hostname you control. Do not copy another operator's hostname, credentials, database, or proxy configuration.

## 1. Install the localhost service

Install Cortex first, then create the authenticated dashboard service:

```bash
HERMES_HOME="$HOME/.hermes" PORT=8100 ./scripts/install_dashboard_service.sh
systemctl --user start cortex-dashboard
systemctl --user status cortex-dashboard --no-pager
```

Save the temporary password printed by the installer. Cortex stores only its salted hash and requires a new password on first sign-in.

The Python server intentionally listens only on `127.0.0.1:8100`. Keep that port closed to the public internet and place a TLS reverse proxy or tunnel in front of it.

## 2. Point a hostname at the host

Create an `A` record for an IPv4 VPS or an `AAAA` record for IPv6:

```text
Type: A
Name: brain
Value: YOUR_VPS_PUBLIC_IP
```

DNS only maps the name. The reverse proxy in the next step terminates HTTPS and forwards requests to Cortex on localhost.

## 3. Choose one HTTPS route

### Caddy

Caddy can obtain and renew a public TLS certificate automatically when ports 80 and 443 reach the VPS:

```caddyfile
brain.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8100
}
```

Reload Caddy after validating the configuration:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

### Nginx

Use a certificate issued for your own hostname. The following is the relevant proxy block, not a complete certificate-installation guide:

```nginx
server {
    listen 443 ssl;
    server_name brain.example.com;

    ssl_certificate /etc/letsencrypt/live/brain.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/brain.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8100;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Cloudflare Tunnel

A tunnel avoids opening an inbound dashboard port. After creating a tunnel and its DNS route, use an ingress entry like:

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /home/YOUR_USER/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: brain.example.com
    service: http://127.0.0.1:8100
  - service: http_status:404
```

Run `cloudflared tunnel run YOUR_TUNNEL_NAME` under a service manager. Cloudflare Access can be added as an outer identity layer; the Cortex password remains the dashboard's own protection.

## 4. Sign in and choose your password

Open `https://brain.example.com`, enter the username and temporary password, then choose a password of at least 12 characters. The **Account** control in the top bar lets you change it later or sign out. A password change revokes older signed sessions.

If access is lost, generate a new temporary password on the host and restart the service:

```bash
PYTHONPATH="$HOME/.hermes/plugins" python3 -m cortex dashboard-password --username cortex
systemctl --user restart cortex-dashboard
```

Use the same username chosen during installation if it is not `cortex`.

## 5. Verify the boundary

The sign-in page itself is public, but the memory data must not be:

```bash
curl -sS https://brain.example.com/api/auth/status
curl -sS -o /dev/null -w '%{http_code}\n' https://brain.example.com/api/snapshot
```

The status response should report `"auth_enabled": true`; an unauthenticated snapshot request should return `401`. Also confirm the site has a valid HTTPS certificate and that the VPS firewall does not expose port `8100`.

## Back up and update

Back up `$HOME/.hermes/cortex/cortex.db` and `$HOME/.hermes/cortex/dashboard-auth.json`. The auth file contains a password hash and session-signing secret, so keep it private and mode `0600`. Never publish either file.

After a Cortex update, rerun `install_local.sh` and restart the dashboard service. The installer preserves an existing auth file and password.
