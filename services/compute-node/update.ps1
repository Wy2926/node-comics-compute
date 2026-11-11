[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Root)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($Root)
$setup = Join-Path $rootPath '.setup'
New-Item -ItemType Directory -Force -Path $setup | Out-Null
$lock = [IO.File]::Open((Join-Path $setup 'update.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    # Run outside the program files being replaced. Keep this helper after an
    # interruption so Update.cmd can recover even if node.exe was moved.
    $helper = Join-Path $setup 'update-host.exe'
    if (-not (Test-Path -LiteralPath (Join-Path $setup 'update/transaction.json'))) {
        Copy-Item -LiteralPath (Join-Path $rootPath 'node.exe') -Destination $helper -Force
    }
    & $helper _update
    $result = $LASTEXITCODE
} finally { $lock.Dispose() }
exit $result
