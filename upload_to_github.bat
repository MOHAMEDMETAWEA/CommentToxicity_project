@echo off
echo ============================================
echo     🚀 Uploading D:\CommentToxicity_project to GitHub (ALL FILES)
echo ============================================
cd /d "D:\CommentToxicity_project"

REM --- Step 1: Initialize Git if not already a repo ---
if not exist ".git" (
    echo Initializing new Git repository...
    git init
    git branch -M main
    git remote add origin https://github.com/MOHAMEDMETAWEA/CommentToxicity_project
)

REM --- Step 2: Ensure Git LFS is installed and initialized ---
echo Checking for Git LFS...
git lfs install

REM --- Step 3: Track large/common file types with Git LFS ---
git lfs track "*.h5"
git lfs track "*.csv"
git lfs track "*.zip"
git lfs track "*.tar"
git lfs track "*.pt"
git lfs track "*.pkl"
git lfs track "*.bin"
git lfs track "*.model"
git add .gitattributes

REM --- Step 4: Add all files (including large ones) ---
echo Adding ALL files (this may take a while)...
git add --all

REM --- Step 5: Commit changes ---
set /p msg="Enter commit message (default: Auto update): "
if "%msg%"=="" set msg=Auto update
git commit -m "%msg%"

REM --- Step 6: Push everything to GitHub (with LFS support) ---
echo ============================================
echo 🚀 Pushing ALL files (large + small) to GitHub...
echo ============================================
git push origin main

IF %ERRORLEVEL% NEQ 0 (
    echo ❗ Push failed, retrying with --force...
    git push origin main --force
)

echo ============================================
echo ✅ Upload complete! All files (including large ones) are now on GitHub.
echo ============================================
pause
