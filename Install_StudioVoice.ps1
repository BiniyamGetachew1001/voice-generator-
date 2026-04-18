<#
.SYNOPSIS
Installs Studio Voice locally by downloading Python (if needed) and all dependencies.
#>
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

# Main GUI Form
$form = New-Object System.Windows.Forms.Form
$form.Text = "Studio Voice Setup"
$form.Size = New-Object System.Drawing.Size(550, 320)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#121212")
$form.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#E0E0E0")

$font = New-Object System.Drawing.Font("Segoe UI", 10)
$form.Font = $font

# Title Label
$titleLabel = New-Object System.Windows.Forms.Label
$titleLabel.Text = "Studio Voice Installer"
$titleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
$titleLabel.Location = New-Object System.Drawing.Point(20, 15)
$titleLabel.Size = New-Object System.Drawing.Size(500, 35)

# Status Label
$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text = "Click 'Install' to begin."
$statusLabel.Location = New-Object System.Drawing.Point(20, 55)
$statusLabel.Size = New-Object System.Drawing.Size(500, 25)

# Progress Bar
$progressBar = New-Object System.Windows.Forms.ProgressBar
$progressBar.Location = New-Object System.Drawing.Point(20, 85)
$progressBar.Size = New-Object System.Drawing.Size(490, 10)
$progressBar.Style = 'Continuous'

# Log View
$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.Location = New-Object System.Drawing.Point(20, 105)
$logBox.Size = New-Object System.Drawing.Size(490, 110)
$logBox.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#1E1E1E")
$logBox.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#A0A0A0")
$logBox.Font = New-Object System.Drawing.Font("Consolas", 9)

# Install Button
$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = "Install"
$btnInstall.Location = New-Object System.Drawing.Point(410, 230)
$btnInstall.Size = New-Object System.Drawing.Size(100, 35)
$btnInstall.BackColor = [System.Drawing.ColorTranslator]::FromHtml("#FF6A1A")
$btnInstall.ForeColor = [System.Drawing.ColorTranslator]::FromHtml("#FFFFFF")
$btnInstall.FlatStyle = 'Flat'
$btnInstall.FlatAppearance.BorderSize = 0

# Add Controls
$form.Controls.Add($titleLabel)
$form.Controls.Add($statusLabel)
$form.Controls.Add($progressBar)
$form.Controls.Add($logBox)
$form.Controls.Add($btnInstall)

$LogMessage = {
    param([string]$msg)
    $logBox.AppendText("[$((Get-Date).ToString("HH:mm:ss"))] $msg`r`n")
    $logBox.SelectionStart = $logBox.Text.Length
    $logBox.ScrollToCaret()
    $statusLabel.Text = $msg
    $form.Refresh()
}

$btnInstall.Add_Click({
    $btnInstall.Enabled = $false
    $progressBar.Style = 'Marquee'
    
    $root = $PSScriptRoot
    
    & $LogMessage "Checking for Windows Python..."
    $pyCheck = Get-Command "python" -ErrorAction SilentlyContinue
    if (-not $pyCheck) {
        & $LogMessage "Python not found. Downloading and installing silently via Winget..."
        $processAction = Start-Process winget -ArgumentList "install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements" -Wait -NoNewWindow -PassThru
        if ($processAction.ExitCode -ne 0 -and $processAction.ExitCode -ne -1978335189) {
            & $LogMessage "Python installation failed. Please install Python manually."
            $progressBar.Style = 'Blocks'
            return
        }
        & $LogMessage "Python installed successfully."
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        $pythonPath = "python"
    } else {
        $pythonPath = $pyCheck.Source
        & $LogMessage "Python found at $pythonPath."
    }

    & $LogMessage "Creating dedicated environment..."
    if (-not (Test-Path "$root\voiceenv")) {
        Start-Process $pythonPath -ArgumentList "-m venv `"$root\voiceenv`"" -Wait -NoNewWindow
    }

    & $LogMessage "Installing AI requirements (this takes a few minutes)..."
    $pipPath = "$root\voiceenv\Scripts\python.exe"
    $reqPath = "$root\requirements.txt"
    Start-Process $pipPath -ArgumentList "-m pip install -r `"$reqPath`" --upgrade" -Wait -NoNewWindow

    & $LogMessage "Creating Desktop Shortcut..."
    $WshShell = New-Object -comObject WScript.Shell
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $Shortcut = $WshShell.CreateShortcut("$DesktopPath\Studio Voice.lnk")
    $Shortcut.TargetPath = "$root\launch_studio_voice.vbs"
    $Shortcut.WorkingDirectory = "$root"
    $Shortcut.Description = "Launch Studio Voice"
    
    # Try to set a default icon
    $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll, 137"
    $Shortcut.Save()
    
    & $LogMessage "Installation Complete!"
    $statusLabel.Text = "Studio Voice is ready to use. Shortcut created on Desktop."
    $progressBar.Style = 'Blocks'
    $progressBar.Value = 100
    
    $btnInstall.Text = "Done"
    $btnInstall.Enabled = $true
    $btnInstall.RemoveAll_Click()
    $btnInstall.Add_Click({ $form.Close() })
})

$form.ShowDialog() | Out-Null
