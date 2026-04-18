Set shell = CreateObject("WScript.Shell")
' Using 1 instead of 0 to show the window for debugging
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\StudioVoice\launch_studio_voice.ps1""", 1, False
