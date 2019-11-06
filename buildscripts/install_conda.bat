echo off

@rem Download and install Python 3.7 x64 Miniconda3 on Windows
@rem Miniconda is taken from https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

@rem Default prefix if %UserProfile%\Miniconda3
@rem To set other prefix use setup_conda.bat <prefix>
set prefix=%UserProfile%\Miniconda3
set installer=%UserProfile%\Miniconda3.exe

echo "Install Miniconda3 on Windows"
IF "%~1"=="" (echo "Note: Conda prefix is empy; To set prefix use setup_conda.bat <prefix>") ELSE (set prefix=%~1)
echo "Conda will be installed to %prefix%"

echo "Remove %prefix% and %installer%"
if exist "%prefix%" RMDIR "%prefix%" /S /Q
if exist "%installer%" del /f "%installer%"

echo "Download Miniconda3 installer to %installer%"
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', '%installer%')"

echo "Install Miniconda3 to %prefix%"
%installer% /InstallationType=JustMe /RegisterPython=0 /S /D=%prefix%
echo "Done"
