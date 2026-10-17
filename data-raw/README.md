# data-raw

Scripts that regenerate the packaged metadata in `src/geniekit/resources/`.
They need network access but no Synapse login; everything they read is public.

| Script | Output | Source |
|---|---|---|
| `build_releases.py` | `releases.json` | Synapse REST API: entity tree of the BPC "Data Releases" folder (syn27056700); cohort facts from aacrprojectgenie.org |
| `build_dictionary.py` | `dictionary.csv.gz`, `dictionary.meta.json` | The six Variable Synopsis workbooks linked from aacrprojectgenie.org (downloaded into `cache/`, which is gitignored) |

Run from the repository root, in this order:

```powershell
python data-raw/build_releases.py
python data-raw/build_dictionary.py
pytest tests/ -q
```

Commit the regenerated resources together with the script change that
produced them; the diff of `releases.json` is the review artefact when a new
public release appears.
