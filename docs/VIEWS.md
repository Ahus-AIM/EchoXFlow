# View labels

Every single-plane B-mode recording carries labels from two published view classifiers, EchoPrime and EchoFine, run
on eight evenly spaced scan-converted frames ([scripts/build_view_classifier_csv.py](../scripts/build_view_classifier_csv.py)). The labels are predictions,
not expert annotations. Each record exposes the top-1 label and probability of both classifiers
(`view_echoprime`, `view_echoprime_prob`, `view_echofine`, `view_echofine_prob`) and a coarse `view` that is set
only when the two agree:

| `view` | EchoPrime label | EchoFine labels |
| --- | --- | --- |
| `A2C` | `A2C` | `a2ch-*` |
| `A3C` | `A3C` | `a3ch-*` |
| `A4C` | `A4C` | `a4ch-*` |
| `PLAX` | `PLAX` | `plax-*`, `mmode-plax-*` |
| `PSAX` | `PSAX` | `psax-*` |
| `Other` | any other | any other |
| `None` | disagreement, or not classified | |

## EchoPrime

[EchoPrime](https://doi.org/10.1038/s41586-025-09850-x) trains an image view classifier on 77,426 sonographer-labelled
echocardiograms, with views defined by the [American Society of Echocardiography guidelines](https://doi.org/10.1016/j.echo.2018.06.004).
The released checkpoint predicts eleven classes. The three Doppler classes are not used because only B-mode
frames are classified. The paper gives no per-class definitions beyond the
guideline names, so the descriptions are the standard meaning of each name.

| label | description |
| --- | --- |
| `A2C` | apical two-chamber |
| `A3C` | apical three-chamber (apical long axis) |
| `A4C` | apical four-chamber |
| `A5C` | apical five-chamber |
| `PLAX` | parasternal long axis (`Parasternal_Long` in the model output) |
| `PSAX` | parasternal short axis (`Parasternal_Short` in the model output) |
| `SSN` | suprasternal notch |
| `Subcostal` | subcostal |

## EchoFine (EchoView47)

[EchoFine](https://doi.org/10.1016/j.media.2026.104006) is trained on TTE47, 91,139 images annotated by a cardiologist into 47 views. The
descriptions below are the paper's own (Appendix A, Tables A.13 to A.19). The grouping follows its Table C.20.
The released weights name one class `doppler-ao-descending` where the paper writes `doppler-ao`.

### Apical

| label | description |
| --- | --- |
| `a2ch-full` | Apical 2-chamber, covering the depth of the whole of the LV and LA. |
| `a2ch-la` | Apical 2-chamber that either has Doppler around the MV or LA, or is plain 2D but not covering the whole depth of the LV. |
| `a2ch-lv` | Apical 2-chamber, covering the whole depth of the LV, but not the whole depth of the LA. |
| `a3ch-full` | Apical 3-chamber, covering the whole depth of the LV and the LA. |
| `a3ch-la` | Apical 3-chamber, that either has Doppler around the MV or LA, or is plain 2D but not covering the whole depth of the LV. |
| `a3ch-lv` | Apical 3-chamber, covering the whole depth of the LV but not the whole depth of the LA. |
| `a3ch-outflow` | Apical 3-chamber, with either colour that is more looking at LVOT than the MV or grayscale but limited to the LVOT area. |
| `a4ch-full` | Apical 4-chamber. Needs to show the full depth of all four chambers without specifically focussing on LV or RV. |
| `a4ch-ias` | Apical 4-chamber, focused on the inter-atrial septum. |
| `a4ch-la` | Apical 4-chamber, that either has Doppler around the MV or LA, or is plain 2D but covering the whole depth of LA and not whole depth of the LV. |
| `a4ch-lv` | Apical 4-chamber, covering the whole depth of the LV and not the whole depth of the LA. |
| `a4ch-ra` | Apical 4-chamber, focused on the right atrium, or with colour doppler on the Tricuspid valve or Right Atrium. |
| `a4ch-rv` | Apical 4-chamber, focused on the right ventricle. |
| `a5ch-full` | Full apical 5-chamber with depth covering from apex to back of atria, without colour on LVOT. |
| `a5ch-outflow` | Apical 5-chamber including the LVOT but either not covering the full depth from apex of the heart to the back of the atria, or with colour on the LVOT. |
| `apex` | Apical window with insufficient depth to reach the mitral ring. |

### Parasternal long axis

| label | description |
| --- | --- |
| `plax-full-la` | Parasternal long-axis, with the imaging sector covering the whole left atrium, intended for LA measurements. |
| `plax-full-lv` | Parasternal long-axis, with the imaging sector covering the LV but not the whole LA. |
| `plax-full-mv` | Parasternal long-axis, centered on the mitral valve. This is signified by either so much zoom that parts of the LA and LV are lost from view or colour doppler over the Mitral Valve. |
| `plax-full-out` | Parasternal long-axis, zoomed out. |
| `plax-full-rv-ao` | Parasternal long-axis, where either the image has been zoomed or otherwise positioned to focus on the right ventricle and aorta or there is colour doppler in the region of the RV, aortic valve, or aorta. |
| `plax-tv` | Parasternal inflow view including the tricuspid valve. |
| `plax-valves-av` | Parasternal long-axis, focussed on the aortic valve. |
| `plax-valves-mv` | Parasternal long-axis, focused on the mitral valve. |

### Parasternal short axis

| label | description |
| --- | --- |
| `psax-all` | Parasternal short-axis, valve level, with the field of view covering the positions of the tricuspid, aortic and pulmonary valves (even if they are not clearly seen). |
| `psax-av` | Parasternal short-axis, with field of view focused on aortic valve. |
| `psax-lv-apex` | Parasternal short-axis, left ventricle apex level. |
| `psax-lv-base` | Parasternal short-axis, left ventricle base level. |
| `psax-lv-mid` | Parasternal short-axis, left ventricle mid-level. |
| `psax-pv` | Parasternal short-axis, focused on pulmonary valve. |
| `psax-tv` | Parasternal short-axis, focused on tricuspid valve. |

### Subcostal and suprasternal

| label | description |
| --- | --- |
| `subcostal-heart` | Subcostal window, focused on the heart. |
| `subcostal-ivc` | Subcostal window, focused on the inferior vena cava. |
| `suprasternal` | Suprasternal view. |

The 13 M-mode, spectral Doppler and tissue Doppler classes are not used, since only B-mode frames are classified.

## Scanner view detector

Recordings also carry the scanner's own ten-class view-detector scores (`native_score_*`, `native_prob_*`). The
class order is not documented by the vendor. `native_class_map.json` in the export gives the inferred meaning of
each class and its agreement with EchoPrime and EchoFine.

## Citation

```bibtex
@article{vukadinovic_2025_echoprime,
  title = {Comprehensive echocardiogram evaluation with view primed vision language AI},
  author = {Vukadinovic, Milos and Chiu, I-Min and Tang, Xiu and Yuan, Neal and Chen, Tien-Yu and Cheng, Paul and Li, Debiao and Cheng, Susan and He, Bryan and Ouyang, David},
  journal = {Nature},
  volume = {650},
  number = {8103},
  pages = {970--977},
  year = {2026},
  doi = {10.1038/s41586-025-09850-x}
}

@article{naidoo_2026_echofine,
  title = {Robust fine-grained echocardiographic view classification with supervised contrastive learning},
  author = {Naidoo, Preshen and Fernandes, Patricia and Dadashi Serej, Nasim and Stowell, Catherine C. and Manisty, Charlotte H. and Francis, Darrel P. and Zolgharni, Massoud},
  journal = {Medical Image Analysis},
  volume = {110},
  pages = {104006},
  year = {2026},
  doi = {10.1016/j.media.2026.104006}
}
```
