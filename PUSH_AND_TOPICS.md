# Push to GitHub and set topics

The repository is ready locally. To publish it on GitHub:

## 1. Create the repo on GitHub

1. Go to [github.com/new](https://github.com/new).
2. Set **Repository name** to: `4dgs-human-avatar-comfyui-nodes-dataset`
3. **Description:** `ComfyUI nodes for 4D Gaussian Splatting human avatars + free and enterprise human scan datasets`
4. Choose **Public**, do **not** add a README (we already have one).
5. Click **Create repository**.

## 2. Push from local

From the repo root (`4dgs-human-avatar-comfyui-nodes-dataset`):

```bash
git remote add origin https://github.com/AndriiShramko/4dgs-human-avatar-comfyui-nodes-dataset.git
git branch -M main
git push -u origin main
```

If your default branch is `master`, use:

```bash
git push -u origin master
```

Use your GitHub username and a [Personal Access Token](https://github.com/settings/tokens) (with `repo` scope) when prompted for password.

## 3. Set repository topics

On the repo page on GitHub: **About** → gear icon → **Topics** → add:

- `4d-gaussian-splatting`
- `comfyui`
- `comfyui-nodes`
- `human-avatar`
- `dataset`
- `neural-rendering`
- `4dgs`
- `3d-reconstruction`

Or run the script (after setting `GITHUB_TOKEN` and creating the repo):

```bash
python scripts/set_repo_topics.py
```

The script is in the repo: run from repo root: `python scripts/set_repo_topics.py` (requires `GITHUB_TOKEN` with repo scope).
