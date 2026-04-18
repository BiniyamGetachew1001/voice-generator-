@echo off
TITLE Studio Voice Installer

:: Check for Administrator privileges (sometimes needed for Winget, but let's just run normally first)
echo Launching Studio Voice Installer Interface...

:: Run the visual installer hidden from command line so only the UI shows
powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0Install_StudioVoice.ps1"

exit
