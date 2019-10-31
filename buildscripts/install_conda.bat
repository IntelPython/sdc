echo on

set prefix=%UserProfile%\Miniconda3
set installer=%UserProfile%\Miniconda3.exe
IF "%1" == "" (echo "Conda prefix is empy; To set prefix use setup_conda.bat <prefix>")
ELSE (set prefix=%1)
echo "Conda will be installed to %prefix%"

if exist "%prefix%" RMDIR "%prefix%" /S /Q
if exist "%installer%" del /f "%installer%"

powershell -Command "(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', '%installer%')"

%installer% /InstallationType=JustMe /RegisterPython=0 /S /D=%prefix%