# infra — Deployment scripts for Google Cloud (probt-cloud)

## deploy.ps1

Full deploy script: builds both images, deploys both Cloud Run services, runs smoke test.

### Prerequisites

- gcloud CLI authenticated (`gcloud auth login`)
- Billing account linked to `probt-cloud`
- APIs enabled (Cloud Run, Artifact Registry, Cloud Build)
- Local engine data generated (`engine/data/*.csv`, `*.pkl`) — see engine README

### Usage

```powershell
# From the repo root
.\infra\deploy.ps1

# Or from infra/
cd infra
.\deploy.ps1
```

### What it does

1. Builds engine Docker image → Artifact Registry
2. Deploys/updates `probt-api` Cloud Run service
3. Captures engine URL
4. Builds web Docker image with engine URL baked in
5. Deploys/updates `probt-web` Cloud Run service
6. Runs smoke test on both services

### Cost

Both services use `--min-instances=0` (scale to zero). Estimated cost for demo traffic: **$0-5/month**.
