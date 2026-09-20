# Third-party software and data

The root `LICENSE` covers EgoANT's original code, prompts, tests, and
documentation. It does **not** relicense third-party software, models, or data.

## Optional hand reconstruction

EgoANT pins [Microsoft VITRA](https://github.com/microsoft/VITRA) at commit
`b35517202b39d32a753fdd42014b2cc3c41fab58` as a Git submodule. VITRA retains
its MIT license and pins its own dependencies, including:

- HaWoR at `891ca88ab5462145241b1336eb3bb1c054aaa494`, whose repository and model
  are distributed under CC BY-NC-ND 4.0;
- MoGe at `286873bdaa16fe05086ac2e081d6bc87b7523118`, under its upstream license;
- transitive components such as DROID-SLAM and Metric3D, under their upstream
  licenses.

The HaWoR and MANO terms can restrict commercial use and redistribution. The
optional hand-reconstruction path is therefore not covered by EgoANT's MIT
license. Review every upstream license before using that path.

## Models and datasets

Model weights are downloaded separately and retain their original terms. The
MANO files are license-gated and are never distributed by this repository.
WGO-Bench, EgoDex, and any input videos also retain their dataset licenses and
are not included in this release.
