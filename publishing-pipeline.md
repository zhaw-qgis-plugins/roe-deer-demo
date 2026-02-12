# Publishing Pipeline

How to publish changes to the Roe Deer Corridor Demo plugin.

## Repositories

| Repo | Purpose |
|------|---------|
| `QGIS-Plugin-Demo/` | Plugin source code |
| `QGIS-Plugin-Repository/` | Static QGIS plugin repository (serves ZIPs + `plugins.xml`) |

## Steps

### 1. Edit the plugin source

Make your changes in `QGIS-Plugin-Demo/roe_deer_corridor_demo/` (e.g. `roe_deer_demo.py`).

### 2. Bump the version

Update the `version` field in `roe_deer_corridor_demo/metadata.txt`:

```ini
version=0.2.0
```

QGIS uses this version string to detect updates. If you don't bump it, users won't see the new release.

### 3. Build and publish to the repository

From `QGIS-Plugin-Demo/`:

```bash
make publish
```

This zips the plugin, copies it to `QGIS-Plugin-Repository/plugins/`, and regenerates `plugins.xml`.

### 4. Commit and push the repository

```bash
git add plugins/roe_deer_corridor_demo.zip plugins.xml
git commit -m "Update Roe Deer Corridor Demo to v0.2.0"
git push
```

Once pushed, GitHub Pages (or whatever serves the repo) will make the new version available. QGIS clients pointing at the repository URL will pick up the update.

## Quick reference

```bash
# From QGIS-Plugin-Demo/
# 1. Edit source + bump version in metadata.txt
# 2. Build and publish
make publish
# 3. Commit and push the repository
cd ../QGIS-Plugin-Repository
git add plugins/roe_deer_corridor_demo.zip plugins.xml
git commit -m "Update Roe Deer Corridor Demo to vX.Y.Z"
git push
```
