[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$IncludeModels,
    [switch]$IncludeFfmpeg,
    [switch]$Archive
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ManifestPath = Join-Path $Root "portable_manifest.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Version = [string]$Manifest.version
if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "portable_manifest.json does not contain version."
}

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $Root "dist"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$Variant = if ($IncludeModels) { "portable" } else { "source" }
$Target = Join-Path $Destination ("DubClean-{0}-{1}" -f $Version, $Variant)

if (Test-Path -LiteralPath $Target) {
    throw "Target directory already exists: $Target"
}

New-Item -ItemType Directory -Path $Target -Force | Out-Null

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$TargetDirectory,
        [string[]]$ExtraExcludedFiles = @()
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required directory is missing: $Source"
    }
    New-Item -ItemType Directory -Path $TargetDirectory -Force | Out-Null
    $excluded = @("*.pyc", "*.pyo", "*.log", "*.tmp", "*.partial") + $ExtraExcludedFiles
    $arguments = @(
        $Source,
        $TargetDirectory,
        "/E",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD",
        "__pycache__",
        ".pytest_cache",
        "/XF"
    ) + $excluded
    & robocopy @arguments | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "Failed to copy $Source (robocopy: $LASTEXITCODE)."
    }
}

$RootFiles = @(
    ".gitattributes",
    ".gitignore",
    "ATTRIBUTION.md",
    "CHECK_INSTALL.bat",
    "COPYRIGHT",
    "INSTALL_DEPS.bat",
    "LICENSE",
    "MODEL_LICENSE.md",
    "NOTICE",
    "portable_manifest.json",
    "PROPRIETARY_LICENSING.md",
    "README.md",
    "requirements.txt",
    "SECURITY.md",
    "SETUP.bat",
    "START_DUBCLEAN.bat",
    "STOP_DUBCLEAN.bat",
    "THIRD_PARTY_NOTICES.md",
    "TRAINING_DATA_DISCLOSURE.md"
)
foreach ($Name in $RootFiles) {
    $Source = Join-Path $Root $Name
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination (Join-Path $Target $Name)
}

foreach ($Directory in @("python_src", "portable_tools", "docs", "licenses")) {
    Copy-Tree (Join-Path $Root $Directory) (Join-Path $Target $Directory)
}
if (-not $IncludeModels) {
    Copy-Tree (Join-Path $Root ".github") (Join-Path $Target ".github")
}

$ModelExclusions = if ($IncludeModels) {
    @()
} else {
    @("last_best_checkpoint.pt", "*.pth", "*.ckpt", "*.onnx", "*.jit")
}
Copy-Tree (Join-Path $Root "models") (Join-Path $Target "models") $ModelExclusions
Copy-Tree (Join-Path $Root "third_party_models") (Join-Path $Target "third_party_models") $ModelExclusions

foreach ($Directory in @("projects", "uploads", "outputs", "runtime", "backups")) {
    New-Item -ItemType Directory -Path (Join-Path $Target "data\$Directory") -Force | Out-Null
}

if ($IncludeFfmpeg) {
    $FfmpegSource = Join-Path $Root "tools\ffmpeg\bin"
    if (-not (Test-Path -LiteralPath (Join-Path $FfmpegSource "ffmpeg.exe"))) {
        throw "tools\ffmpeg\bin\ffmpeg.exe is missing."
    }
    Copy-Tree $FfmpegSource (Join-Path $Target "tools\ffmpeg\bin")
}

$Checker = Join-Path $Target "portable_tools\check_release.py"
$CheckerArguments = @($Checker, $Target)
if ($IncludeModels) {
    $CheckerArguments += "--require-models"
}
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
& $Python @CheckerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Release validation failed."
}

if ($Archive) {
    $ArchivePath = "$Target.zip"
    if (Test-Path -LiteralPath $ArchivePath) {
        throw "Archive already exists: $ArchivePath"
    }
    Compress-Archive -LiteralPath $Target -DestinationPath $ArchivePath -CompressionLevel Optimal
    Write-Host "Archive ready: $ArchivePath"
}

Write-Host "Release ready: $Target"
