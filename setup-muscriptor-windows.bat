@echo off
setlocal EnableExtensions DisableDelayedExpansion
REM ============================================================================
REM  setup-muscriptor-windows.bat
REM
REM  One-shot, idempotent setup of MuScriptor on a Windows 11 x64 machine with
REM  an NVIDIA GPU, fully GPU-ready (CUDA PyTorch). Re-running is safe:
REM  finished steps are skipped and downloads resume or hit local caches.
REM
REM  Steps:
REM    1. Verify an NVIDIA driver supporting CUDA 12.8+ (required by the cu128
REM       PyTorch wheels; RTX 4080/4090 and newer GeForce GPUs are covered).
REM    2. Ensure git is installed (installed via winget if missing).
REM    3. Ensure uv is installed (uv 0.9.26 installed if missing).
REM    4. Clone https://github.com/muscriptor/muscriptor and check out the
REM       pinned, tested commit.
REM    5. Apply the Windows CUDA fix: PyPI only ships CPU-only torch wheels
REM       for Windows, so pin torch to PyTorch's official cu128 wheel index
REM       in pyproject.toml, relock, and verify uv.lock got a "+cu128" torch.
REM    6. uv sync (downloads the ~3 GB CUDA torch wheel once), then assert
REM       torch.cuda.is_available() and run a real matmul on the GPU.
REM    7. HuggingFace auth (weights are license-gated) and pre-download of
REM       the muscriptor-large weights (5.46 GB, cached, resumable).
REM    8. End-to-end self-test: synthesize 2 s of audio and transcribe it with
REM       "--model large --device cuda" - fails loudly if the GPU path breaks.
REM
REM  Usage:  setup-muscriptor-windows.bat [target-dir]
REM          default target-dir: .\muscriptor under the current directory
REM
REM  Tested on: RTX 4090 Laptop 16 GB, driver 572.83 (CUDA 12.8), Win11 x64,
REM  July 2026. Written for a 16 GB Ada GPU (RTX 4080/4090); any NVIDIA GPU
REM  with a CUDA 12.8-capable driver works.
REM ============================================================================

set "REPO_URL=https://github.com/muscriptor/muscriptor"
set "PINNED_COMMIT=6c1460cc75e5f120948de7656da05b2c489e8715"
set "UV_INSTALLER=https://astral.sh/uv/0.9.26/install.ps1"

set "TARGET_DIR=%CD%\muscriptor"
if not "%~1"=="" set "TARGET_DIR=%~f1"

REM UTF-8-safe Python I/O; silence the harmless HF cache symlink warning
REM (non-admin Windows cannot create symlinks; hf falls back to plain copies).
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

echo.
echo === MuScriptor GPU setup ===
echo Target directory: "%TARGET_DIR%"
echo.

REM ---------------------------------------------------------------- 1. GPU --
echo [1/8] Checking NVIDIA GPU and driver...
where /q nvidia-smi
if errorlevel 1 (
    echo [FAIL] nvidia-smi not found - no NVIDIA driver installed.
    echo        Install the GeForce driver from https://www.nvidia.com/drivers
    echo        then reboot and re-run this script.
    goto :fail
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

set "CUDAVER_FILE=%TEMP%\muscriptor_cudaver.txt"
powershell -NoProfile -Command "$m = (nvidia-smi) | Select-String 'CUDA Version:\s*([0-9]+)\.([0-9]+)'; if ($m) { Write-Output ($m.Matches[0].Groups[1].Value + ' ' + $m.Matches[0].Groups[2].Value) }" > "%CUDAVER_FILE%"
set "CUDA_MAJOR="
set "CUDA_MINOR="
for /f "usebackq tokens=1,2" %%a in ("%CUDAVER_FILE%") do (set "CUDA_MAJOR=%%a" & set "CUDA_MINOR=%%b")
del /q "%CUDAVER_FILE%" >nul 2>nul
if not defined CUDA_MAJOR (
    echo [FAIL] Could not parse the CUDA version from nvidia-smi output.
    goto :fail
)
echo Driver supports CUDA %CUDA_MAJOR%.%CUDA_MINOR%
if %CUDA_MAJOR% GTR 12 goto :gpu_ok
if %CUDA_MAJOR% EQU 12 if %CUDA_MINOR% GEQ 8 goto :gpu_ok
echo [FAIL] The cu128 PyTorch wheels need a driver supporting CUDA 12.8+,
echo        but this driver only supports CUDA %CUDA_MAJOR%.%CUDA_MINOR%.
echo        Update the NVIDIA driver, reboot, then re-run this script.
goto :fail
:gpu_ok

REM ---------------------------------------------------------------- 2. git --
echo.
echo [2/8] Checking git...
set "GIT=git"
where /q git
if errorlevel 1 goto :install_git
goto :git_ok
:install_git
echo git not found - installing Git for Windows via winget...
where /q winget
if errorlevel 1 (
    echo [FAIL] Neither git nor winget is available. Install git manually from
    echo        https://git-scm.com/download/win and re-run this script.
    goto :fail
)
winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
set "GIT=%ProgramFiles%\Git\cmd\git.exe"
if not exist "%GIT%" (
    echo [FAIL] git still missing after winget install. Install it manually from
    echo        https://git-scm.com/download/win and re-run this script.
    goto :fail
)
:git_ok
"%GIT%" --version

REM ----------------------------------------------------------------- 3. uv --
echo.
echo [3/8] Checking uv...
set "UV=uv"
where /q uv
if errorlevel 1 goto :install_uv
goto :uv_ok
:install_uv
echo uv not found - installing uv 0.9.26 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm %UV_INSTALLER% | iex"
set "UV=%USERPROFILE%\.local\bin\uv.exe"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if not exist "%UV%" (
    echo [FAIL] uv installation did not produce uv.exe. Install manually:
    echo        https://docs.astral.sh/uv/getting-started/installation/
    goto :fail
)
:uv_ok
"%UV%" --version
if errorlevel 1 (
    echo [FAIL] uv is present but not runnable.
    goto :fail
)

REM ---------------------------------------------------- 4. clone + checkout --
echo.
echo [4/8] Getting the repo at pinned commit %PINNED_COMMIT:~0,7% ...
if exist "%TARGET_DIR%\.git" goto :repo_present
if exist "%TARGET_DIR%" goto :target_conflict
"%GIT%" clone %REPO_URL% "%TARGET_DIR%"
if errorlevel 1 (echo [FAIL] git clone failed - check network access. & goto :fail)
goto :repo_ready
:target_conflict
echo [FAIL] Target directory exists but is not a git repo: "%TARGET_DIR%"
echo        Delete it or pass a different target directory as argument 1.
goto :fail
:repo_present
echo Repo already present - reusing it.
:repo_ready
"%GIT%" -C "%TARGET_DIR%" checkout --detach %PINNED_COMMIT% >nul 2>nul
if not errorlevel 1 goto :checkout_ok
"%GIT%" -C "%TARGET_DIR%" fetch origin
"%GIT%" -C "%TARGET_DIR%" checkout --detach %PINNED_COMMIT%
if errorlevel 1 (echo [FAIL] Cannot check out the pinned commit. & goto :fail)
:checkout_ok
"%GIT%" -C "%TARGET_DIR%" log -1 --oneline
pushd "%TARGET_DIR%"

REM -------------------------------------------------- 5. CUDA fix + uv lock --
echo.
echo [5/8] Applying Windows CUDA fix and locking dependencies...
findstr /C:"pytorch-cu128" pyproject.toml >nul 2>nul
if not errorlevel 1 goto :patch_done
>>pyproject.toml echo(
>>pyproject.toml echo # --- Windows CUDA fix, appended by setup-muscriptor-windows.bat ---
>>pyproject.toml echo # PyPI only publishes CPU-only torch wheels for Windows. Resolve torch from
>>pyproject.toml echo # the official PyTorch CUDA 12.8 index on Windows instead; other platforms
>>pyproject.toml echo # keep resolving torch from PyPI.
>>pyproject.toml echo [tool.uv.sources]
>>pyproject.toml echo torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
>>pyproject.toml echo(
>>pyproject.toml echo [[tool.uv.index]]
>>pyproject.toml echo name = "pytorch-cu128"
>>pyproject.toml echo url = "https://download.pytorch.org/whl/cu128"
>>pyproject.toml echo explicit = true
echo Patched pyproject.toml.
:patch_done
"%UV%" lock
if errorlevel 1 (echo [FAIL] uv lock failed - check network access. & goto :fail)
findstr /C:"+cu128" uv.lock >nul
if errorlevel 1 (echo [FAIL] uv.lock contains no +cu128 torch - the CUDA pin did not take. & goto :fail)
echo CUDA torch confirmed in uv.lock:
findstr /C:"version = " uv.lock | findstr /C:"+cu128"

REM ------------------------------------------------- 6. install + GPU check --
echo.
echo [6/8] Installing the environment - first run downloads the ~3 GB CUDA torch wheel...
"%UV%" sync
if errorlevel 1 (echo [FAIL] uv sync failed. & goto :fail)
"%UV%" run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; x = torch.randn(256, 256, device='cuda'); y = (x @ x).sum().item(); print('torch', torch.__version__, '- GPU OK:', torch.cuda.get_device_name(0))"
if errorlevel 1 (echo [FAIL] torch cannot use the GPU. If the driver was just updated, reboot and re-run. & goto :fail)

REM ------------------------------------------------- 7. HF auth + weights ---
echo.
echo [7/8] HuggingFace authentication and model pre-download...
if defined HF_TOKEN goto :hf_authed
if defined HF_HOME if exist "%HF_HOME%\token" goto :hf_authed
if exist "%USERPROFILE%\.cache\huggingface\token" goto :hf_authed
echo.
echo No HuggingFace token found on this machine. The MuScriptor weights are
echo license-gated, so you need a free HuggingFace account that has ACCEPTED
echo the model license:
echo   1. Log in on https://huggingface.co and accept the license at
echo      https://huggingface.co/MuScriptor/muscriptor-large
echo      - one click, access is granted automatically.
echo   2. Create a READ token at https://huggingface.co/settings/tokens
echo      and paste it at the prompt below.
echo.
"%UV%" run hf auth login
if errorlevel 1 (echo [FAIL] HuggingFace login failed. & goto :fail)
:hf_authed
"%UV%" run hf auth whoami
if errorlevel 1 (echo [FAIL] HuggingFace token found but not valid - run: uv run hf auth login & goto :fail)
echo Downloading muscriptor-large weights - 5.46 GB, skipped when already cached...
"%UV%" run hf download MuScriptor/muscriptor-large config.json model.safetensors
if errorlevel 1 (
    echo [FAIL] Weight download failed. Most common cause: the license at
    echo        https://huggingface.co/MuScriptor/muscriptor-large has not been
    echo        accepted by the account this machine is logged in as.
    goto :fail
)

REM ------------------------------------------------------- 8. GPU self-test --
echo.
echo [8/8] End-to-end self-test - loads the 1.4B model onto the GPU, ~1-2 min...
set "ST_WAV=%TEMP%\muscriptor-selftest.wav"
set "ST_MID=%TEMP%\muscriptor-selftest.mid"
"%UV%" run python -c "import numpy as np, soundfile as sf, os; sr = 32000; t = np.arange(2 * sr) / sr; wav = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)) * np.exp(-1.5 * t); sf.write(os.environ['ST_WAV'], wav.astype('float32'), sr)"
if errorlevel 1 (echo [FAIL] Could not synthesize the test audio file. & goto :fail)
"%UV%" run muscriptor transcribe "%ST_WAV%" -o "%ST_MID%" --model large --device cuda
if errorlevel 1 (echo [FAIL] GPU transcription failed. & goto :fail)
if not exist "%ST_MID%" (echo [FAIL] Transcription reported success but produced no MIDI file. & goto :fail)
del /q "%ST_WAV%" "%ST_MID%" >nul 2>nul
popd

echo.
echo ============================================================
echo  SUCCESS - MuScriptor is fully set up and GPU-verified.
echo ============================================================
echo.
echo  Repo: "%TARGET_DIR%"
echo.
echo  Use it like this:
echo    cd /d "%TARGET_DIR%"
echo    uv run muscriptor transcribe "C:\path\to\song.mp3" -o song.mid --model large
echo.
echo  Notes:
echo    - The large model peaks near 16 GB VRAM at the default GPU batch size.
echo      If you hit CUDA out-of-memory, add:  -b 2
echo    - Weights live in %%USERPROFILE%%\.cache\huggingface, the repo venv in .venv.
echo    - Re-running this script is safe; completed steps are skipped.
echo.
pause
exit /b 0

:fail
echo.
echo Setup aborted. Fix the issue above and re-run - completed steps are skipped.
popd >nul 2>nul
pause
exit /b 1
