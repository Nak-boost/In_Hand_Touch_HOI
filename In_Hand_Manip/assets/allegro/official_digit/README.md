# Sourced DIGIT / Allegro geometry

See [assembly documentation](../../../docs/official_digit.md) for the exact
mounting profile, geometry changes, collision approximations, and validation.

- `source_manifest.json`: upstream URLs, pinned revisions, SHA256 and archive provenance.
- `source/`: unmodified selected upstream files and their license notices.
- `visual/`: partition of TACTO's published DIGIT surface into housing and curved gel.
- `collision/`: derived convex approximations, not original CAD or soft-body geometry.
- `mesh_manifest.json`: generated geometry hashes and V-HACD parameters.

Attribution: DIGIT by Lambeta et al.; TACTO by Wang et al. / Meta (Facebook)
Research; Allegro by SimLab / Wonik Robotics; NeuralFeels / FeelSight by Suresh
et al. Upstream licenses remain applicable to their respective files:

- DIGIT hardware STEP/adapter STL: CC-BY-NC 4.0, `source/digit_LICENSE`.
- TACTO: MIT, `source/tacto_LICENSE`; Allegro example description: BSD,
  `source/tacto_hand_LICENSE`.
- Vendor Allegro: BSD, `source/vendor_LICENSE`.
- FeelSight: MIT as declared in the preserved `source/feelsight_README.md`.

These files are not relicensed under the repository's top-level license.
No independently guessed adapter transform, mesh rescaling, or hardware mass
calibration is implied by the package name. Derived inertia is an explicit
uniform-envelope approximation using the original paper's nominal sensor mass.
