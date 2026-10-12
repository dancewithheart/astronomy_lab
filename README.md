# Learning ML in Python on astronomy data

![ML algorithm decision map](ml_roadmap.svg)

## Pleiades: recover a star cluster (clustering - DBSCAN, HDBSCAN)

```sh
python -m experiments.pleiades --refresh
```

Approach:
```
query Gaia dataset
-> pandas cleaning/filtering
-> feature selection
-> StandardScaler
-> run skylearn unsupervised learning - DBSCAN
-> visualise candidate cluster and data (results + noise): motion, sky, CMD (colour–magnitude sequence)
```

Results:
Gaia query returned: 720 sources (stars)
DBSCAN identified a single dense group
```
392 stars
median parallax = 7.361 mas
median pmra     = +19.749 mas/yr
median pmdec    = -45.370 mas/yr
```

From parallax distance is 1000 / 7.36 = 136 pc


* Proper-motion plot

![Proper-motion plot](reports/pleiades/proper-motion-clusters.png)

The yellow DBSCAN cluster is a very compact blob around roughly (pmra, pmdec) ≈ (20, -45) while the field stars are widely dispersed. This is exactly the structure we hoped unsupervised clustering would recover.

* Sky plot

![Sky plot](reports/pleiades/sky-clusters.png)

The yellow candidates are spread across most of 1° cone rather than forming an obvious dense central patch.
That looked like something worth investigate further:

![radial-density analysis](reports/pleiades/radial-density.png)
![CDF analysis](reports/pleiades/angular-distance-cdf.png)

* CMD

![CMD](reports/pleiades/cmd-candidate-vs-field.png)

Cluster 0 forms a nice stellar sequence from hot/blue bright stars
down to cool/red faint stars.

That is nice evidence supporting claim that DBSCAN did not find accidental data.

### Pleiades - parameters tuning for DBSCAN

```
python -m experiments.pleiades_tune
```

After dropping conditions on parallex there are more data 4k sources
so the experiment become bore general (no bias towards parallex).

But across different parameters the same cluster is found:
```
parallax ≈ 7.36 mas
pmra     ≈ +19.72 mas/yr
pmdec    ≈ -45.32 mas/yr
```

After varying eps and in the best combination was eps=0.05 and min_samples=15
```
eps=0.05
min_samples=15

candidate size: 962
TP: 858
FP: 104
FN: 23

precision: 0.892
recall:    0.974
F1:        0.931
```
it improves initial parameter choice:
```
precision: 0.712 → 0.892
recall:    1.000 → 0.974
F1:        0.832 → 0.931
```

In data there is visible *precision–recall trade-off*
```
0.05 / 10 → precision .874, recall .984
0.05 / 15 → precision .892, recall .974   ← best F1
0.05 / 20 → precision .910, recall .936
0.05 / 30 → precision .916, recall .900
0.05 / 40 → precision .927, recall .846
...
```

### Compare ML models HDBSCAN vs DBSCAN

HDBSCAN was more robust to parameter choice but more permissive:
it consistently recovered all 881 reference members,
while including ~325–333 additional candidates.

Tuned DBSCAN sacrificed 23 reference members but reduced unmatched candidates to 104, producing a higher F1 score.

```
DBSCAN:
precision = 0.892
recall    = 0.974
F1        = 0.931

HDBSCAN:
precision = 0.731
recall    = 1.000
F1        = 0.844
```

### Research using this approach

* 2023 study used Gaia DR3 + DBSCAN on Pleiades, Praesepe and Blanco 1; it identified 958 Pleiades members: [A Gaia astrometric view of the open clusters Pleiades, Praesepe and Blanco 1 - Jeison Alfonso, Alejandro García-Varela - Astrometry & Astrophysics Vol 677](https://www.aanda.org/articles/aa/full_html/2023/09/aa46569-23/aa46569-23.html)
* 2024 study used Gaia DR3 + DBSCAN to twelve open clusters (NGC 2264, NGC 2682, NGC 2244, NGC 3293, NGC 6913, NGC 7142, IC 1805, NGC 6231, NGC 2243, NGC 6451, NGC 6005, NGC 6583) and validated the resulting memberships with colour-magnitude diagrams and spectroscopic data [Membership determination in open clusters using the DBSCAN Clustering Algorithm - Mudasir Raja, Priya Hasan, Md Mahmudunnobe, Md Saifuddin, S N Hasan](https://arxiv.org/abs/2404.10477)


## Pleiades: recover a star cluster

Find unusual objects in the colour–magnitude diagram (CMD)

```sh
python -m experiments.gaia_anomalies
```

Features:
* `bp_rp` - stellar colour, related to temperature
* `absolute_g_mag` - intrinsic brightness, estimated from parallax

```python
FEATURES = ["bp_rp", "absolute_g_mag"]
```

Use two methods:
* `IsolationForest` uses randomly generated decision trees. Observations that can be isolated with relatively few splits receive higher anomaly scores.
* `Local Outlier Factor` (LOF) asks a different question: is this point unusually isolated relative to its local neighbours?

![distance distribution](reports/gaia_anomalies/isolation-forest-cmd.png)

Three particularly interesting regions:
- Bottom-left: blue, intrinsically faint objects forming a sequence. These are consistent with the white-dwarf cooling sequence.
- Bottom-right: very red, intrinsically faint stars, plausibly low-mass M dwarfs and ultracool objects.
- Top-right: extremely luminous, red objects, potentially evolved giant stars, heavily reddened objects, or problematic measurements.

IsolationForest identifies entire rare populations, not necessarily individual objects.

![lof](reports/gaia_anomalies/lof-cmd.png)

LOF tends to highlight stars that deviate from the local CMD structure, including objects between the main stellar sequences.



### verification using SIMBAD

```
python -m experiments.gaia_anomalies
```

Use AQL query
```sql
SELECT
    u.gaia_id,
    b.main_id,
    b.otype,
    b.otype_txt
FROM TAP_UPLOAD.candidates AS u
JOIN ident AS i
    ON i.id = u.gaia_id
JOIN basic AS b
    ON b.oid = i.oidref
```
to [query SIMBAD using astroquery](https://astroquery.readthedocs.io/en/latest/api/astroquery.simbad.SimbadClass.html) to check what is type of candidate anomalies
```sh
python -m experiments.gaia_anomaly_validate
```

| Column | Meaning |
|---|---|
| `source_id` | Gaia DR3 identifier |
| `method` | IsolationForest or LOF |
| `if_rank` | IsolationForest anomaly ranking |
| `lof_rank` | Local Outlier Factor ranking |
| `main_id` | SIMBAD's preferred object identifier |
| `otype` | Astronomical object-type code |
| `otype_txt` | Human-readable object classification |
| `simbad_found` | Whether an identifier match was found |

Summary:

| Metric | IsolationForest | LOF |
|---|---:|---:|
| Candidates | 200 | 200 |
| Matched in SIMBAD | **195 (97.5%)** | 88 (44%) |
| Confirmed white dwarfs (`WD*`) | 65 | 3 |
| White-dwarf candidates (`WD?`) | 23 | 2 |
| White dwarfs + candidates | **88 (44%)** | 5 (2.5%) |
| No exact SIMBAD match | 5 | 112 |

Example:

| Gaia DR3 source | IF rank | LOF rank | SIMBAD classification |
|---|---:|---:|---|
| [69975466908268416](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+69975466908268416&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id) | 1 | 124 | White-dwarf candidate (`WD?`) |
| [64061503103122048](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+64061503103122048&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id) | 7 | 28,730 | White dwarf (`WD*`) |
| [69886853143994496](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+69886853143994496&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id) | 240 | 1 | Hot subdwarf (`HS*`) |

IsolationForest considers the white dwarf extremely unusual; LOF considers it entirely ordinary.
Why might that happen?
Because white dwarfs occupy a relatively distinct region of the CMD, yet form their own sequence. A star can therefore be globally unusual while locally surrounded by similar stars.

### Research

* [Gaia Data Release 3, Apsis. III. Non-stellar content and source classification](https://www.aanda.org/articles/aa/full_html/2023/06/aa43423-22/aa43423-22.html), Delchambre et al., 2022; Process 56 mln of sources using SOM.
* [Astronomaly: Personalised Active Anomaly Detection in Astronomical Data](https://arxiv.org/abs/2010.11202), Michelle Lochner, Bruce A. Bassett, 2021; combines anomaly detection with `active learning`. Tool using this approach: https://github.com/MichelleLochner/astronomaly

## Pleiades from Gaia using dimensionality reduction (PCA, UMAP)

```sh
python -m experiments.gaia_dimensionality --features 3
```

PCA and UMAP project Gaia measurements into two dimensions:
* `PCA` - linear projection, preserving maximum variance.
* `UMAP` - nonlinear projection, preserving local neighbourhoods.

Both reveal a distinct Pleiades population, verified against published members.

On UMAP there is small, detached group in the upper-right.

![umap](reports/gaia_dimensionality/3d/umap.png)

![pca](reports/gaia_dimensionality/3d/pca.png)

Validation and compare 3 vs 5 features:

```sh
python -m experiments.gaia_dimensionality --features 5
python -m experiments.gaia_dimensionality --features 3
python -m experiments.gaia_dimensionality_validate
```

Results:

| Method | 3 features | 5 features | Better |
|---|---:|---:|---|
| PCA, published-member 10NN | **86.9%** | 79.3% | 3D |
| UMAP, published-member 10NN | 86.4% | **89.5%** | 5D |
| PCA, proxy 10NN | **98.0%** | 84.5% | 3D |
| UMAP, proxy 10NN | **98.4%** | 92.8% | 3D |

PCA performs better with three features.
UMAP performs slightly better with five features against the published catalogue.

Here `10NN` measures `local neighbourhood purity`.

So **adding features does not necessarily improve a machine learning result. Their relevance depends on the algorithm and the objective**.

## Orion constellation - data exploration and visualisation

```sh
python -m experiments.orion.analysis
python -m experiments.orion.analysis --refresh
```

Gets astronomy data from Gaia DR3 data source about stars around [Orion](https://en.wikipedia.org/wiki/Orion_(constellation)):
- distance distribution

![distance distribution](reports/orion/analysis/distance-histogram.png)

- color vs distance

![color vs distance](reports/orion/analysis/distance-vs-color.png)

- parallax measurement quality

![parallax measurement quality](reports/orion/analysis/parallax-quality.png)

```sh
python -m experiments.orion.pyvista --mode screenshot --quality draft
python -m experiments.orion.pyvista --mode screenshot --quality final
python -m experiments.orion.pyvista --mode mp4 --quality final
python -m experiments.orion.pyvista --mode interactive --quality preview
```

on Ubuntu, PyVista needed OpenGL libraries:

```bash
sudo apt-get install -y libgl1 libglx-mesa0 libegl1
```

Renders those Orion constellation in 3D using:
- distances computed from parallax
- Gaia BP−RP colours
or open interactive mode.

The current nebula dust is artistic - Gaussian blobs around bright stars and AI-tuned warm/cool gradients.

![syntetic Orion visualisation](reports/orion/gaia_orion_pyvista.png)


Interactive Plotly visualization (should open on http://127.0.0.1:43701/):

```bash
python -m experiments.orion.flythrough
```

![Interactive Orion visualisation](reports/orion/plotly_visualisation.png)
