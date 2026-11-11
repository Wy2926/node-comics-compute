# Shared developer/portable bootstrap; only the dedicated work directory is written.
function Initialize-NodeTools([string]$LockFile, [string]$WorkDirectory) {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    $toolchain = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
    $downloads = Join-Path $WorkDirectory 'downloads'
    $run = Join-Path $WorkDirectory ('run-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Force -Path $downloads,$run | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    function Get-FileDigest([string]$path) {
        $stream = [IO.File]::OpenRead($path)
        $hash = [Security.Cryptography.SHA256]::Create()
        try { return [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
        finally { $stream.Dispose(); $hash.Dispose() }
    }
    function Get-VerifiedArchive($asset, [string]$name) {
        $target = Join-Path $downloads $name
        if (-not (Test-Path -LiteralPath $target)) {
            Write-Host "Downloading pinned $name"
            $request = [Net.HttpWebRequest]::Create($asset.url)
            $request.Timeout = 120000
            $request.ReadWriteTimeout = 120000
            $response = $request.GetResponse()
            try {
                $inputStream = $response.GetResponseStream()
                $outputStream = [IO.File]::Create($target + '.partial')
                try {
                    $buffer = New-Object byte[] 1048576
                    $received = 0L
                    $timer = [Diagnostics.Stopwatch]::StartNew()
                    while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                        $outputStream.Write($buffer, 0, $count)
                        $received += $count
                        if ($timer.ElapsedMilliseconds -ge 1000) {
                            $detail = '{0:N1} MiB' -f ($received / 1MB)
                            if ($response.ContentLength -gt 0) { $detail += ' / {0:N1} MiB ({1:N0}%)' -f ($response.ContentLength / 1MB), (100 * $received / $response.ContentLength) }
                            Write-Host "  ${name}: $detail"
                            $timer.Restart()
                        }
                    }
                    Write-Host ('  {0}: {1:N1} MiB downloaded; verifying SHA-256' -f $name, ($received / 1MB))
                } finally { $inputStream.Dispose(); $outputStream.Dispose() }
            } finally { $response.Dispose() }
            if ((Get-FileDigest ($target + '.partial')) -ne $asset.sha256) { throw "Download checksum mismatch: $name" }
            Move-Item -LiteralPath ($target + '.partial') -Destination $target
        }
        Write-Host "  Verifying cached $name"
        if ((Get-FileDigest $target) -ne $asset.sha256) { throw "Cached archive checksum mismatch: $name" }
        return $target
    }
    $uvArchive = Get-VerifiedArchive $toolchain.uv 'uv.zip'
    $pythonArchive = Get-VerifiedArchive $toolchain.python 'python.tar.gz'
    Write-Host '[2/6] Extracting private Python and package tools...'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $uvRoot = Join-Path $run 'uv'
    $zip = [IO.Compression.ZipFile]::OpenRead($uvArchive)
    try {
        foreach ($entry in $zip.Entries) {
            $path = [IO.Path]::GetFullPath((Join-Path $uvRoot $entry.FullName))
            if (-not $path.StartsWith($uvRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe tool archive entry.' }
        }
    } finally { $zip.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($uvArchive, $uvRoot)
    $tar = Join-Path $env:SystemRoot 'System32/tar.exe'
    if (-not (Test-Path -LiteralPath $tar)) { throw 'Windows tar.exe is required (Windows 10 1803 or newer).' }
    $entries = & $tar -tzf $pythonArchive
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Python archive.' }
    foreach ($entry in $entries) {
        if (-not $entry.StartsWith('python/') -or $entry -match '(^|/)\.\.(/|$)' -or $entry.Contains('\')) { throw 'Unsafe Python archive entry.' }
    }
    & $tar -xzf $pythonArchive -C $run
    if ($LASTEXITCODE -ne 0) { throw 'Cannot extract pinned Python runtime.' }
    $uv = Join-Path $uvRoot 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) { throw 'Pinned uv archive has unexpected layout.' }
    return @{ Run = $run; Python = (Join-Path $run 'python/python.exe'); Uv = $uv }
}
