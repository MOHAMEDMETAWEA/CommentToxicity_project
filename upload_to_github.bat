@echo off
echo ============================================
echo     🚀 Uploading D:\CommentToxicity_project to GitHub
echo ============================================
cd /d D:\CommentToxicity_project

REM Initialize Git if not already a repo
if not exist ".git" (
    echo Initializing new Git repository...
    git init
    git branch -M main
    git remote add origin https://github.com/MOHAMEDMETAWEA/CommentToxicity_project
)

REM Add all files
echo Adding files...
git add .

REM Commit changes
set /p msg="Enter commit message (default: Auto update): "
if "%msg%"=="" set msg=Auto update
git commit -m "%msg%"

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main --force

echo ============================================
echo ✅ Upload complete! Check your GitHub repo.
echo ============================================
pause
