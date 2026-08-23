param(
    [string]$PgRoot = "C:\Program Files\PostgreSQL\18",
    [string]$PgVectorVersion = "v0.8.6"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command nmake -ErrorAction SilentlyContinue)) {
    Write-Error "nmake was not found. Run this from 'x64 Native Tools Command Prompt for VS 2022' as Administrator."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git was not found on PATH."
}

if (-not (Test-Path -LiteralPath $PgRoot)) {
    Write-Error "PostgreSQL root not found: $PgRoot"
}

$env:PGROOT = $PgRoot
$workDir = Join-Path $env:TEMP "pgvector-build"
$repoDir = Join-Path $workDir "pgvector"

New-Item -ItemType Directory -Force -Path $workDir | Out-Null

if (Test-Path -LiteralPath $repoDir) {
    Remove-Item -LiteralPath $repoDir -Recurse -Force
}

git clone --branch $PgVectorVersion https://github.com/pgvector/pgvector.git $repoDir
Push-Location $repoDir
try {
    nmake /F Makefile.win
    nmake /F Makefile.win install
}
finally {
    Pop-Location
}

Write-Host "pgvector $PgVectorVersion installed into $PgRoot"

