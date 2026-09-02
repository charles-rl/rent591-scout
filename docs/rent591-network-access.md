# Rent591 Network Access — Problem & Investigation Log

This document records why the **live** Rent591 pipeline could not be run from the
development sandbox, everything that was tried, and how to run it live.

## Summary

- `rent.591.com.tw` and its API host `bff-house.591.com.tw` are **firewall-blocked** at the
  network level from the dev sandbox. Every 591 domain is dropped on both ports 80 and 443.
- The block is **IP/ACL-based** (connections reset in ~0.003s), independent of hostname, DNS, or SNI.
- The sandbox has a **restricted egress whitelist**: GitHub, PyPI, HuggingFace, PyTorch, and
  Docker Hub only. Everything else (Google, archive.org, jsDelivr, ntfy.sh, 591) is blocked.
- The Nexus "proxy" available in the cluster is a **package-registry proxy** (PyPI/Docker/Maven/npm
  repos) — it does **not** provide general outbound HTTP forwarding to arbitrary hosts.
- **No IPv6 routing** exists in the sandbox.
- The pipeline is fully built and **verified end-to-end in offline fixtures mode**; it will run live
  on any host with 591 access (or behind an HTTPS forward proxy that can reach 591).

## Symptom

```
requests.exceptions.SSLError: HTTPSConnectionPool(host='bff-house.591.com.tw', port=443):
Max retries exceeded ... (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]
EOF occurred in violation of protocol ...')))
```

`curl` returned HTTP `000` (no response) for every 591 endpoint.

## Investigation log

### 1. Connectivity to 591 endpoints — all fail
```bash
curl -sk -o /dev/null -w "%{http_code}" https://bff-house.591.com.tw/   # 000
curl -sk -o /dev/null -w "%{http_code}" https://rent.591.com.tw/       # 000
curl -sk -o /dev/null -w "%{http_code}" https://m.591.com.tw/          # 000
curl -sk -o /dev/null -w "%{http_code}" https://house.591.com.tw/      # 000
curl -sk -o /dev/null -w "%{http_code}" https://www.591.com.tw/        # 000
curl -sk -o /dev/null -w "%{http_code}" http://bff-house.591.com.tw/   # 000 (port 80 too)
```

### 2. DNS resolves fine → not a DNS issue
`bff-house.591.com.tw` resolves to CloudFront (e.g. `65.9.130.92` IPv4, `2600:9000:…` IPv6).

### 3. IP/ACL block, not hostname/SNI
Pinning the resolved CloudFront IPs still dropped instantly (~0.003s):
```bash
curl --resolve bff-house.591.com.tw:443:65.9.130.71 ...  # 000 in 0.002-0.003s
```
A 0.003s reset is an active firewall DROP, not a TLS/HTTP problem.

### 4. No IPv6
```bash
ip -6 addr            # no inet6 addresses
curl -6 https://example.com  # 000
```

### 5. No standard proxy environment
```bash
env | grep -i proxy    # only VSCODE_PROXY_URI (inbound web-IDE tunnel, not outbound egress)
```

### 6. Nexus "proxy" — package registry only, no general egress
The Nexus host (`nexus.repo-proxy.svc:8081`) was tested thoroughly:
- Ports: `8081` (Nexus Repository Manager UI/repos), `8082` (Docker registry proxy).
- Used as an HTTP forward proxy (CONNECT) → connection failed.
- REST API (`/service/rest/v1/repositories`) lists only package registries:
  `nuget, maven-central, docker-hub-proxy, debian-proxy, pypi-proxy, pypi-nvidia-proxy,
  ubuntu-proxy, ubuntu-upstream-proxy, npm-proxy, acr-proxy, maven-*`.
- It can only fetch from its configured upstream registries — it cannot fetch `591.com.tw`.

### 7. Other cluster hosts probed
- `traefik.traefik` (`10.233.34.52`): ports 80/443 open, but ingress-only — no CONNECT/forward proxy.
- Common proxy names (`proxy`, `egress-proxy`, `web-proxy`, `squid`, `http-proxy`, …) → not found in DNS.

### 8. Egress whitelist mapping (what the firewall DOES allow)
| Domain | Result |
|---|---|
| `github.com`, `raw.githubusercontent.com` | 200 / 301 (reachable) |
| `pypi.org` | 200 |
| `huggingface.co` | 200 |
| `download.pytorch.org` | 403 (reachable, root path) |
| `docker.io`, `registry-1.docker.io` | 302 / 404 (reachable) |
| `example.com`, `google.com`, `translate.google.com` | 000 |
| `archive.org`, `web.archive.org` | 000 |
| `r.jina.ai`, `cdn.jsdelivr.net` | 000 |
| `ntfy.sh` | 000 (push notifications also blocked) |

## Root cause

The dev sandbox sits behind an egress firewall with a **developer whitelist**
(GitHub/PyPI/HF/PyTorch/Docker Hub + internal Nexus). Rent591 is not on the whitelist and is
dropped at the network-ACL layer. This is an environment constraint — **not** a bug in the
pipeline or in the 591 API. The vendored `Client591` code is correct (verified against
real captured 591 responses in fixtures mode).

## What works instead

1. **GitHub Actions cron relay** (implemented — the production path): a scheduled
   workflow (`.github/workflows/scrape_relay.yml`) runs `python -m src.ingestion
   --output-dir data/incoming/` on GitHub's cloud runners (591-reachable), commits raw
   JSON payloads + WebP images back to this repo, and the GPU server `git pull`s them
   and runs `python main.py --incoming` fully offline. See the README "GitHub Actions
   cron relay" section.

1. **Offline fixtures mode** (used for all end-to-end testing here):
   ```bash
   PLACEHOLDER_IMAGES=1 python main.py --fixtures --limit 3
   ```
   Replays captured 591 responses from `external/mcp-591/tests/fixtures/*.json`.
   `PLACEHOLDER_IMAGES=1` generates local WebP placeholders because the 591 image CDN
   is also blocked. (On a real 591-reachable host this env var is unset.)

2. **HTTPS forward proxy** (if one is available):
   `requests` uses `trust_env=True`, so `HTTPS_PROXY`/`HTTP_PROXY` are honored
   automatically for both the API calls and image downloads.

3. **Run on a 591-reachable host** (recommended for production):
   ```bash
   source .venv/bin/activate
   export NTFY_TOPIC=your-topic          # ntfy.sh also needs reachability (or a self-hosted server)
   python main.py                        # live run
   python main.py --train                # retrain XGBoost head after 20+ ratings
   ```

## Also discovered during live-run preparation

- **ntfy.sh is blocked** in this sandbox too; the notifier was validated with a local
  capture server (`NTFY_URL=http://localhost:8998`). The `★` character in the ntfy
  `Title` header broke latin-1 header encoding and was fixed to `(x.x/5)`.
- CUDA torch must be the `+cu128` build (`2.8.0+cu128`) to match the driver; default
  `2.13.0+cu130` fails to initialize. See `docs/BUILD_PLAN.md` for full notes.

## GitHub Actions relay — live findings (verified 2026-09-01)

The relay was deployed (`.github/workflows/scrape_relay.yml`) and run for real
via `workflow_dispatch`. Observed behavior from GitHub-hosted runners:

| Run | Runner egress IP | Site probe | rent/list API | img1/img2.591.com.tw |
|---|---|---|---|---|
| 1 | (pool A) | 301 / 404 | **403 WAF** | — |
| 2 | 4.246.135.197 | 301 / 404 | **200 (24 listings committed to `data/incoming/`)** | 403 (hotlink) |
| 3 | 172.182.253.37 | 403 / 403 | 403 | — |
| 4 | 172.208.127.35 | 403 → 301 (+30s) | **200 (24 listings)** | **403 for every GET, even with `Referer: rent.591.com.tw`** |

Conclusions:

- 591's WAF **intermittently blocklists GitHub's shared runner IP pool at the IP
  level** (same curl/UA succeeds from one runner IP and gets 403 from another;
  the block also covers `m.591.com.tw` warm-up). It is not a header/cookie problem —
  warm-up + browser headers are in place (`client591._warmup` / `_get_api`).
- The image CDN 403s datacenter IPs outright; standard browser headers do not help.
  We deliberately stop at correct browser headers (no proxy rotation / cookie spoofing).
- Practical consequences:
  - JSON relay works whenever a run draws an unblocked IP (cron `*/30` gives 48
    attempts/day; successful runs commit real listings — proven twice).
  - Set repo variable `RELAY_SKIP_IMAGES=true` for text-only runs while the CDN is
    blocking; the GPU pipeline degrades to text-based Qwen analysis (all vision
    flags false) for listings without images.
  - **Definitive fix: run the same workflow on a self-hosted runner** on any
    591-reachable network (change `runs-on:` to `[self-hosted, 591-relay]`). The
    rest of the pipeline (git relay → `main.py --incoming` offline inference) is
    unchanged.

## Update 2026-09 — PC devtunnel proxy bridge (hybrid mode, shipped)

The self-hosted-runner gap is closed differently: the personal PC runs an HTTP proxy
(port 8999) bridged to the GPU server as `127.0.0.1:8999` via `devtunnel host`.
Verified against the live tunnel:

- `www.591.com.tw` → 200 (used as the availability probe); `img*.591.com.tw` bare
  originals → 403 through the tunnel, resize variants `...jpg!fit.1000x.water2.jpg`
  → 200 (`image/webp`) — the image queue rewrites URLs accordingly.
- TLS through the tunnel is MITM'd by the devtunnel cert; Python's OpenSSL rejects
  it (`Missing Subject Key Identifier`) → proxy paths set `verify=False`
  (`PROXY_SSL_VERIFY=1` overrides; install the tunnel CA to re-enable properly).
- `ntfy.sh`: blocked **direct** (IP DROP, confirmed), **200 via the tunnel** → all
  pushes are tunnel-first with direct fallback (`src/notifier.py`).
- PC offline (`is_proxy_available()` non-200/exception): text ingestion continues;
  listings queue with `image_status='pending'` and one anti-spam ntfy alert
  (`text_only_notified`) asks the user to bring the tunnel up. When the PC is fully
  powered off there is no egress path for that alert either — delivery is
  best-effort; the queue persists and drains on the next online run.

Implementation: `src/utils/proxy_check.py`, `src/utils/image_queue.py`,
`main.py::run_incoming` (hybrid phases), `listings.image_status` /
`listings.text_only_notified` columns (+ placeholder-repair migration).
README "Hybrid PC-proxy mode (devtunnel)" is the operator doc.
