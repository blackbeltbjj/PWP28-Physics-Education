# Reproducibility assessment — PWP28

## Scope

This repository provides scientific code, derived daily Pacific Warm Pool area data, figures, and supporting documentation for the PWP28 time-series analysis.

The canonical sea-surface temperature threshold is 28 °C. The derived daily series covers 1 September 1981 through 30 September 2026 (16,466 observations).

## Independent execution

Programs RBEF04–RBEF09 were executed in a separate audit directory using a copy of the repository inputs. The original scientific project was not modified.

All 17 corresponding CSV tables matched the original scientific outputs, using absolute and relative numerical tolerances of 1e-10. The comparison included time-series summaries, annual trends, climatology, residuals, autocorrelation diagnostics, hindcast evaluation, and forecasts.

The numerical comparison report is included in this directory.

## Figures

Manuscript Figures 2–9 were regenerated. Visual inspection found no apparent substantive differences.

The regenerated PNG files have larger pixel dimensions than the manuscript copies. Resized image comparisons showed pixel correlations between approximately 0.922 and 0.997. These image metrics support visual consistency but do not, by themselves, establish numerical equivalence.

The original submitted figures and higher-resolution reproductions are stored separately under figures/.

## Figure 1 limitation

The complete spatial reproduction of Figure 1 requires the original NOAA OISST v2.1 gridded inputs and the Pacific-domain mask. Those spatial inputs are not included in the derived daily-area dataset.

The presence of the Figure 1 program and documentation should not be interpreted as an independent end-to-end reproduction from raw gridded SST in this audit.

## Interpretation

This assessment establishes computational agreement between the archived scientific outputs and an isolated rerun using the supplied derived inputs. It does not independently validate the underlying observational dataset, spatial masking decisions, physical interpretations, or forecasting skill beyond the reported evaluation procedure.

## Audit records

- NUMERICAL_REPRODUCTION_COMPARISON.csv
- FIGURE_PIXEL_COMPARISON.csv
- FIGURE_RESIZED_COMPARISON.csv

Release preparation: October 2026.