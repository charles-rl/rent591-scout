# Rent591 Network Access — Findings & Live Operation Notes

Condensed from the original investigation log (pre-relay era). The full curl-probe /
Nexus-registry test trail was removed; every conclusion below is unchanged.

## Summary

- `rent.591.com.tw` and its API host `bff-house.591.com.tw` are **firewall-blocked** at the
  network level from this GPU dev sandbox: every 591 domain drops on ports 80/443 in ~0.003s
  (active firewall DROP — IP/ACL-based, not DNS/SNI; no IPv6 routing exists here).
- Sandbox egress is a **developer whitelist**: GitHub, PyPI, HuggingFace, PyTorch, Docker Hub
  (+ the internal Nexus package-registry proxy, which does **not** forward arbitrary HTTP).
  Everything else — `ntfy.sh`, Google, archive.org, cdn.jsdelivr.net — is blocked.
- Root cause: environment constraint, not a pipeline or 591 API bug. The vendored
  `Client591` code was verified correct against captured live responses in fixtures mode.

## What works (production path)

1. **GitHub Actions cron relay** (tier schedule in `docs/591research.md` §5): runners scrape and
   commit raw payloads + WebP to `data/incoming/`; the GPU server ingests offline via
   `main.py --incoming`. See README "GitHub Actions cron relay".
2. **Offline fixtures mode** for end-to-end testing:
   ```bash
   PLACEHOLDER_IMAGES=1 python main.py --fixtures --limit 3
   ```
3. **PC devtunnel proxy bridge (hybrid mode, shipped):** images + ntfy flow through the
   personal PC's HTTP proxy at `127.0.0.1:8999`. Operator doc: README "Hybrid PC-proxy mode";
   code map (`proxy_check` / `image_queue` / relay idempotency) in AGENTS.md.

## 591 WAF/CDN behavior from GitHub runners (live findings, verified 2026-09-01)

| Run | Runner egress IP | Site probe | rent/list API | img1/img2.591.com.tw |
|---|---|---|---|---|
| 1 | (pool A) | 301 / 404 | **403 WAF** | — |
| 2 | 4.246.135.197 | 301 / 404 | **200 (24 listings committed to `data/incoming/`)** | 403 (hotlink) |
| 3 | 172.182.253.37 | 403 / 403 | 403 | — |
| 4 | 172.208.127.35 | 403 → 301 (+30s) | **200 (24 listings)** | **403 for every GET, even with `Referer: rent.591.com.tw`** |

- 591's WAF **intermittently blocklists GitHub's shared runner IP pool at the IP level**:
  the same curl/UA succeeds from one runner and gets 403 from another (the block also covers
  `m.591.com.tw` warm-up). It is not a header/cookie problem — warm-up + browser headers are in
  place (`client591._warmup` / `_get_api`).
- The image CDN 403s datacenter IPs outright; standard browser headers do not help (we stop at
  correct browser headers — no proxy rotation / cookie spoofing). Through the PC tunnel, bare
  originals also 403 → request resize variants instead (`PROXY_IMAGE_SUFFIX`, see README).
- Practical consequences:
  - JSON relay works whenever a run draws an unblocked IP (three cron tiers give ~100 attempts/
    day; successful runs commit real listings — proven twice above).
  - Repo variable `RELAY_SKIP_IMAGES=true` switches to text-only relay while the CDN blocks;
    the GPU pipeline degrades to text-based Qwen analysis for listings without images.
  - The deterministic fix is a self-hosted runner on any 591-reachable network
    (`runs-on: [self-hosted, 591-relay]`); for this setup the hybrid tunnel bridge (above)
    supersedes it.

## Misc (discovered during live-run preparation)

- `ntfy.sh` is blocked **direct** in this sandbox; the notifier was validated with a local
  capture server and all pushes are now tunnel-first with direct fallback (`src/notifier.py`).
- CUDA torch must be the `2.8.0+cu128` wheel build to match the driver (default cu130 fails to
  initialize).
