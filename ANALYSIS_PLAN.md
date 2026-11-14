# Analysis plan and dated amendments

The plan was kept under version control; dates are those of the commits in the authors' working repository.

## Pre-specified plan (fixed 14 October 2026, before any outcome model was fitted)

- **Population:** adults (≥18 years) with an isolated CABG admission reimbursed by the SUS, admitted 2008–2025.
- **Technique:** billed procedure code (on-pump vs off-pump); number of grafts used only for description and stratification.
- **Case mix (all models):** age (natural cubic spline, 4 df), sex, admission type (elective vs urgent), principal diagnosis (6 groups), year fixed effects.
- **Hospital characteristics (models without hospital fixed effects):** region and legal status (modal value 2012–2025).
- **Estimands:** (1) patient level, technique vs in-hospital death; (2) hospital strategy, annual hospital OPCAB proportion excluding the index patient, per +10 percentage points and in categories fixed in advance (0%, >0–5%, >5–15%, >15–30%, >30%), between hospitals (primary) and within hospitals (hospital fixed effects, sensitivity).
- **Scales and inference:** odds ratios from logistic models fitted to aggregated covariate patterns; risk differences by marginal standardization; standard errors clustered by hospital; wild cluster score bootstrap (Rademacher weights, 9,999 replications) for the primary coefficients.
- **Between-hospital variation:** random-intercept logistic models (median odds ratio, latent ICC), concentration indices and year-to-year stability.
- **Sensitivity analyses:** 2008–2022; elective admissions only; hospital-years with ≥10 CABG; excluding 2020–2021; death or transfer; hospitals with stable graft coding; influence of the most heavily weighted hospitals in fixed-effects models.
- **Exploratory:** instrumental variable analysis using the prior-year hospital OPCAB proportion.

## Amendment 1 (15 October 2026, before any model including 2003–2007 was fitted)

Reason: new data (records from the pre-2008 procedure tables), not results.
- Adoption and between-hospital variation: 2003–2025.
- Mortality: primary analysis unchanged (2008–2025 with admission type); 2003–2025 and 2008–2025 without admission type added as sensitivity analyses, because admission type and discharge-status coding are not comparable before 2008.

## Amendment 2 (15 October 2026, after the results had been seen)

The authors considered 2003–2025 without admission type as the primary mortality analysis, because the admission-type distribution was clinically implausible. In the submitted manuscript the pre-specified analysis (2008–2025 with admission type) is primary and 2003–2025 is reported as a sensitivity analysis; all windows are shown side by side (Supplementary Table S9).

## Post hoc analyses (19 October–5 November 2026)

- 19 October: exploratory analysis of non-SUS admissions (CIHA).
- 21–23 October: predetermined and shrunken exposures; reliability of the annual proportion; event study of sustained changes (definitions fixed before outcomes were examined); volume, years since the first CABG and state; any vs no OPCAB activity; restriction to hospitals that identify the within-hospital model; patient-level association by hospital OPCAB category and graft code; timing of death and alternative outcomes; coding and payment series; conversion bias analysis based on published values; hospital bootstrap of fixed-effects risk differences; models without principal diagnosis.
- 5 November: within–between (Mundlak) model; event study with post-change years grouped as 0–1 and 2–5; conversion bias analysis by hospital OPCAB category; all post hoc analyses repeated in the pre-specified specification.

No correction for multiplicity was applied to post hoc analyses.
