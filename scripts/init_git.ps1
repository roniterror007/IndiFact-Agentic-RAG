param(
    [string]$DefaultBranch = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    git init
}

git checkout -B $DefaultBranch

$hasCommit = $true
try {
    git rev-parse --verify HEAD | Out-Null
} catch {
    $hasCommit = $false
}

if (-not $hasCommit) {
    git -c user.name="Local User" -c user.email="local@example.com" commit --allow-empty -m "Initialize repository" | Out-Null
}

git checkout -B dev
git checkout $DefaultBranch

Write-Host "Repository initialized with branches: $DefaultBranch, dev"
Write-Host "Create feature branches with: git checkout -b feature/<name> dev"
