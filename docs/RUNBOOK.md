# Travely API — Operations Runbook

Last updated: 2026-05-21

## Services

| Service | Platform | Health check |
|---------|----------|--------------|
| Frontend (`travely-app`) | Vercel | App loads, `/auth` reachable |
| API (`travely-backend`) | Render | `GET /health` → `{"status":"healthy"}` |

## Environment matrix

| Variable | Dev | Staging | Production |
|----------|-----|---------|------------|
| `ENV` | `development` | `staging` | `production` |
| `AUTH_DISABLED` | `true` (optional) | `false` | `false` |
| `DOCS_ENABLED` | `true` | `true` | `false` |
| `SENTRY_DSN` | optional | required | required |
| `SENTRY_RELEASE` | git SHA | git SHA | git SHA |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | optional | required | required |

## Deploy checklist

### Backend (Render)

1. Set env vars from `.env.example` (never commit secrets).
2. Set `DOCS_ENABLED=false` in production.
3. Set `AUTH_DISABLED=false` and `GOOGLE_APPLICATION_CREDENTIALS_JSON`.
4. Use paid tier or min instances = 1 to reduce cold starts (target P95 < 2s warm).
5. Verify: `curl https://<api>/health`

### Frontend (Vercel)

1. Set all `VITE_*` vars per environment.
2. Set `VITE_SENTRY_DSN` and `VITE_APP_VERSION` (git SHA) for release tracking.
3. Set `VITE_API_BASE_URL` to production Render URL.
4. Verify signup → quiz → dashboard flow.

### Firestore

```bash
cd travely-app
firebase deploy --only firestore:rules
```

Test rules in staging before production.

## CI/CD

| Repo | Trigger | Pipeline |
|------|---------|----------|
| `travely-app` | PR + push to `main` | lint → build |
| `travely-backend` | PR + push to `main` | ruff → pytest |

Production deploys: tag release or merge to `main` (Render/Vercel auto-deploy if configured).

## Monitoring

- **Sentry**: errors + traces for frontend and API; filter by `release` and `environment`.
- **Render**: enable health check path `/health`.
- **Logs**: production API emits JSON logs with `request_id`, `latency_ms`, `uid_hash`.

## Incident response

1. Check Sentry for new issues (sort by frequency).
2. Check Render service status and recent deploys.
3. Hit `/health` and one authenticated `/recommendations` call.
4. If Firestore errors: verify rules deploy and Firebase status page.
5. Roll back Render/Vercel to last known good deploy if needed.
6. Post-mortem: link Sentry issue, deploy SHA, and root cause.

## Load smoke test

With API running locally (`AUTH_DISABLED=true`):

```bash
python scripts/load_smoke.py --base-url http://127.0.0.1:8000 --requests 20 --concurrency 5
```

With k6 installed:

```bash
k6 run tests/load/recommendations_smoke.js -e BASE_URL=http://127.0.0.1:8000
```

## SLA notes (Render)

- Free tier: cold starts may exceed 2s P95 — not suitable for production.
- Paid tier with min instances = 1 recommended for launch.
- Document expected cold-start behavior in status communications.

## Legal URLs

Configure in Vercel:

- `VITE_TERMS_URL`
- `VITE_PRIVACY_URL`

Used on signup footer in the frontend.
