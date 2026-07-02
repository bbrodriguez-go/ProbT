<#
.SYNOPSIS
  Full deploy of probt (engine + web) to Google Cloud Run.

.DESCRIPTION
  1. Builds engine Docker image and pushes to Artifact Registry
  2. Deploys probt-api Cloud Run service
  3. Captures engine URL
  4. Updates web/cloudbuild.yaml with engine URL
  5. Builds web Docker image
  6. Deploys probt-web Cloud Run service
  7. Smoke tests both services

.PARAMETER Project
  GCP project ID. Default: probt-cloud

.PARAMETER Region
  GCP region. Default: us-central1
#>

param(
  [string]$Project = "probt-cloud",
  [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"
$Repo = "$Region-docker.pkg.dev/$Project/probt"
$RepoWeb = "$Region-docker.pkg.dev/$Project/probt/web:latest"
$RepoEngine = "$Region-docker.pkg.dev/$Project/probt/engine:latest"
$Root = Resolve-Path "$PSScriptRoot\.."

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  probt deploy -> $Project ($Region)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ─── 1. Build engine ─────────────────────────────────────────────
Write-Host "`n[1/7] Building engine image..." -ForegroundColor Yellow
gcloud builds submit "$Root\engine" --tag=$RepoEngine --project=$Project --quiet
if ($LASTEXITCODE -ne 0) { throw "Engine build failed" }

# ─── 2. Deploy engine ────────────────────────────────────────────
Write-Host "`n[2/7] Deploying probt-api..." -ForegroundColor Yellow
gcloud run deploy probt-api `
  --image=$RepoEngine `
  --region=$Region --port=8080 --allow-unauthenticated `
  --memory=1Gi --cpu=1 --min-instances=0 --max-instances=3 `
  --timeout=300 --no-cpu-boost --project=$Project --quiet
if ($LASTEXITCODE -ne 0) { throw "Engine deploy failed" }

# ─── 3. Capture engine URL ───────────────────────────────────────
$ApiUrl = (gcloud run services describe probt-api --region=$Region `
  --format="value(status.url)" --project=$Project).Trim()
$WsUrl = $ApiUrl -replace "^https://", "wss://" -replace "^http://", "ws://"
$WsUrl = "$WsUrl/ws/stream"
Write-Host "`n[3/7] Engine URL: $ApiUrl" -ForegroundColor Green
Write-Host "       WS URL:     $WsUrl" -ForegroundColor Green

# ─── 4. Update cloudbuild.yaml ───────────────────────────────────
Write-Host "`n[4/7] Updating web/cloudbuild.yaml..." -ForegroundColor Yellow
$cloudbuild = @"
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--build-arg'
      - 'API_URL=$ApiUrl'
      - '--build-arg'
      - 'WS_URL=$WsUrl'
      - '-t'
      - '$RepoWeb'
      - '.'
images:
  - '$RepoWeb'
options:
  logging: CLOUD_LOGGING_ONLY
"@
Set-Content "$Root\web\cloudbuild.yaml" $cloudbuild -NoNewline

# ─── 5. Build web ────────────────────────────────────────────────
Write-Host "`n[5/7] Building web image..." -ForegroundColor Yellow
gcloud builds submit "$Root\web" --config="$Root\web\cloudbuild.yaml" --project=$Project --quiet
if ($LASTEXITCODE -ne 0) { throw "Web build failed" }

# ─── 6. Deploy web ───────────────────────────────────────────────
Write-Host "`n[6/7] Deploying probt-web..." -ForegroundColor Yellow
gcloud run deploy probt-web `
  --image=$RepoWeb `
  --region=$Region --port=3000 --allow-unauthenticated `
  --memory=512Mi --cpu=1 --min-instances=0 --max-instances=3 `
  --timeout=60 --cpu-boost --project=$Project --quiet
if ($LASTEXITCODE -ne 0) { throw "Web deploy failed" }

$WebUrl = (gcloud run services describe probt-web --region=$Region `
  --format="value(status.url)" --project=$Project).Trim()

# ─── 7. Smoke test ───────────────────────────────────────────────
Write-Host "`n[7/7] Smoke testing..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
try {
  $h = (Invoke-WebRequest -Uri "$ApiUrl/api/health" -UseBasicParsing -TimeoutSec 60).Content
  Write-Host "  Engine health: $h" -ForegroundColor Green
} catch {
  Write-Host "  Engine cold start, retrying in 20s..." -ForegroundColor DarkGray
  Start-Sleep -Seconds 20
  $h = (Invoke-WebRequest -Uri "$ApiUrl/api/health" -UseBasicParsing -TimeoutSec 60).Content
  Write-Host "  Engine health: $h" -ForegroundColor Green
}
try {
  $w = Invoke-WebRequest -Uri $WebUrl -UseBasicParsing -TimeoutSec 60
  Write-Host "  Web status: $($w.StatusCode)" -ForegroundColor Green
} catch {
  Write-Host "  Web cold start, retrying in 15s..." -ForegroundColor DarkGray
  Start-Sleep -Seconds 15
  $w = Invoke-WebRequest -Uri $WebUrl -UseBasicParsing -TimeoutSec 60
  Write-Host "  Web status: $($w.StatusCode)" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DEPLOY COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Dashboard: $WebUrl" -ForegroundColor White
Write-Host "  API:       $ApiUrl" -ForegroundColor White
Write-Host "  Docs:      $ApiUrl/docs" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
