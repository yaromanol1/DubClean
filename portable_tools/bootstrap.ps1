[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$PackageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$LocalPython = Join-Path $PackageRoot "tools\python311\python.exe"
try {
    $Available = Test-Path -LiteralPath $LocalPython
    if (-not $Available) {
        foreach ($Candidate in @(@("py", "-3.11"), @("python"))) {
            $Executable = $Candidate[0]
            if (Get-Command $Executable -ErrorAction SilentlyContinue) {
                $Arguments = @($Candidate | Select-Object -Skip 1)
                & $Executable @Arguments -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32 else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) { $Available = $true; break }
            }
        }
    }
    if (-not $Available) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $DownloadDirectory = Join-Path $PackageRoot ".downloads"
        New-Item -ItemType Directory -Path $DownloadDirectory -Force | Out-Null
        $Installer = Join-Path $DownloadDirectory "python-3.11.9-amd64.exe"
        Write-Host "[DubClean] Downloading Python 3.11 x64 from python.org..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $Installer
        $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
        if ($Signature.Status -ne "Valid" -or $Signature.SignerCertificate.Subject -notmatch "Python Software Foundation") {
            throw "The Python installer does not have a valid Python Software Foundation signature."
        }
        $PythonDirectory = Split-Path -Parent $LocalPython
        $Process = Start-Process -FilePath $Installer -ArgumentList @("/quiet", "InstallAllUsers=0", "TargetDir=`"$PythonDirectory`"", "Include_launcher=0", "PrependPath=0", "Shortcuts=0", "Include_doc=0", "Include_test=0") -WindowStyle Hidden -Wait -PassThru
        if ($Process.ExitCode -ne 0 -and $Process.ExitCode -ne 3010) { throw "Python installer failed: $($Process.ExitCode)" }
        if (-not (Test-Path -LiteralPath $LocalPython)) { throw "Python installation did not create python.exe" }
    }
    $env:DUBCLEAN_NO_PAUSE = "1"
    & (Join-Path $PackageRoot "INSTALL_DEPS.bat")
    exit $LASTEXITCODE
} catch {
    Write-Host "[DubClean] Setup failed: $_"
    exit 1
}
