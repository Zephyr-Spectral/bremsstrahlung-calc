# Dance et al. (1968) Experimental Data

## Source
W.E. Dance, D.H. Rester, B.J. Farmer, J.H. Johnson, and L.D. Baggerly,
"Bremsstrahlung Produced in Thick Aluminum and Iron Targets by 0.5 to 2.8 MeV Electrons,"
J. Appl. Phys. 39, 2881-2889 (1968).

## Conditions
- **Materials**: Al (Z=13), Fe (Z=26)
- **Electron energies**: 0.5, 1.0, 2.0, 2.8 MeV
- **Detection angles**: 0, 10, 20, 30, 45, 60, 75, 120, 150 deg
- **Target thickness**: CSDA range (thick targets)
  - Al 2 MeV: 1.20 g/cm^2
  - Fe 0.5 MeV: 0.257 g/cm^2
  - Fe 1.0 MeV: 0.613 g/cm^2
  - Fe 2.8 MeV: 2.31 g/cm^2
- **Uncertainty**: 15-18% per data point, 11% on integrated radiated intensity

## Units
- Intensity: k * dn/dk/dOmega [MeV / (MeV sr electron)]
- Same units as NASA TN D-4755 and our calculated spectra

## Digitization
Data points digitized from Pandola et al. (2015), arXiv:1410.2002, Figs. 1-3.
These are reproductions of the original Dance et al. plots.
Digitization tool: WebPlotDigitizer (https://automeris.io/wpd/)
Digitization uncertainty: < 5% (per Pandola et al.)

## Files
- `dance_1968.json` — All digitized data in structured JSON
- `fig1_al_2mev_angles.png` — Pandola Fig 1 (for digitization)
- `fig2_fe_multi_energy.png` — Pandola Fig 2 (for digitization)
- `fig3_al_fe_integrated.png` — Pandola Fig 3 (for digitization)
- `wpd_exports/` — Raw WebPlotDigitizer CSV exports (one per curve)
