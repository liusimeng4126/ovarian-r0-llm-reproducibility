# GitHub upload instructions

Do not upload from this agent. After creating an empty GitHub repository, run:

```powershell
cd "ovarian_r0_llm_public_v1.0.1"
git init
git add .
git status
git commit -m "Manuscript reproducibility release v1.0.0"
git branch -M main
git remote add origin <USER_GITHUB_REPOSITORY_URL>
git push -u origin main
```

GitHub should contain the unpacked repository files. The ZIP is for local archival, Zenodo upload, or manuscript-release preservation and should not be the sole GitHub repository content.
