Add-Type -AssemblyName System.Windows.Forms

$root = 'C:\StudioVoice'
$python = Join-Path $root 'voiceenv\Scripts\python.exe'
$script = Join-Path $root 'studio_voice.py'
$log = Join-Path $root 'studio_voice_server.log'
$err = Join-Path $root 'studio_voice_server.err.log'

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTORCH_NO_CUDA_MEMORY_CACHING = '1'
$env:PHONEMIZER_ESPEAK_LIBRARY = Join-Path $root 'voiceenv\Lib\site-packages\espeakng_loader\espeak-ng.dll'
$env:ESPEAK_DATA_PATH = Join-Path $root 'voiceenv\Lib\site-packages\espeakng_loader\espeak-ng-data'

function Test-StudioVoice {
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:7860/' -UseBasicParsing -TimeoutSec 2
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-StudioVoiceProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -like '*C:\StudioVoice\studio_voice.py*'
    }
}

if (Test-StudioVoice) {
    Start-Process 'http://127.0.0.1:7860/'
    exit 0
}

if (-not (Test-Path $python)) {
    [System.Windows.Forms.MessageBox]::Show('Studio Voice Python environment was not found.', 'Studio Voice') | Out-Null
    exit 1
}

$existing = @(Get-StudioVoiceProcesses)
if (-not $existing) {
    Start-Process -FilePath $python -ArgumentList ('"' + $script + '"') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err | Out-Null
}

for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 750
    if (Test-StudioVoice) {
        Start-Process 'http://127.0.0.1:7860/'
        exit 0
    }
}

$stale = @(Get-StudioVoiceProcesses)
if ($stale.Count -gt 1) {
    $stale | Sort-Object ProcessId | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

Start-Process 'http://127.0.0.1:7860/'
