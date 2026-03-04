# Shaping NYC College & Career Readiness

## Model Deployment

<div align='center'>

Our full model, with included dashboard and project metrics, is uploaded on Streamlit.app! Click below to view the results of our Beta Regression model, as well as adjust our features to view predicted College & Career Readiness (CCR) outcomes.

<a href="https://collegeandcareerdashboard.streamlit.app/Predictive_Tool" target="_blank">
  <img src="https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live App"/>
</a>

</div>

## Critical Research Question (CRQ)
> **To what extent do a high school’s Economic Need Index (ENI) and Percent of Students in Temporary Housing predicts its College and Career Readiness (CCR), how do these environmental stressors impact specific demographic subgroups within the same schools,and which schools serve as "positive outliers" by defying these systemic predictors?**

## Key Actionable Recommendations
* **Recommendation 1 (Policy):** The NYC DOE should direct supplemental CCR funding to schools in the top ENI quartile (≥ 0.85), where a single standard-deviation increase in economic need is associated with a **−16.4 % point drop** in college readiness. Priority should be given to schools with co-occurring housing instability (≥ 5 % temporary housing) such as (**schools example**), which compounds the poverty effect by an additional **−4.6 % points**.
* **Recommendation 2 (Resource):** Invest in **attendance intervention programs** at high-need schools (>0.80). Attendance is the strongest *modifiable* predictor in the model — a 5-percentage-point improvement (~1 SD) is associated with **+4.7 % points** higher CCR. Unlike ENI and housing status, attendance is partially within reach of school-level action (e.g., mentoring, early-warning systems, family outreach).
* **Recommendation 3 (Data/Tech):** Lower the DOE's **n < 15 suppression threshold** or adopt differential-privacy techniques for subgroup CCR reporting. Currently, 76 % of Asian and 80 % of White subgroup CCR data is suppressed, creating a systematic visibility gap that inflates apparent racial disparities and prevents equitable resource allocation. A city-level data-sharing agreement could enable privacy-preserving subgroup analysis without compromising individual student confidentiality.

## Document Navigation

- [Methodology & Technical Specifications](#methodology--technical-specifications)
  - [Data Sources](#data-sources)
  - [Tools Utilized](#tools-utilized)
  - [Modeling Approach](#modeling-approach)
  - [Model Results](#model-results)
    - [Standardized Beta Coefficients (Logit scale) with 95% CI](#standardized-beta-coefficients-logit-scale-with-95ci)
    - [Back-transformed Coefficients (1-SD Shift on CCR % Scale)](#back-transformed-coefficients-effect-of-a-1-sd-shift-on-the-ccr-percentage-scale-evaluated-at-the-intercept-baseline-of-543-)
  - [Model Diagnostic](#model-diagnostic)
- [Key Findings](#key-findings)
- [Links to Final Deliverables](#links-to-final-deliverables)
- [Data Source Attribution](#data-source-attribution)
- [Contributor and Roles](#contributor-and-roles)
- [Tree Structure](#tree-structure)


## Methodology & Technical Specifications
<div align="center">

| | |
| :---: | :---: |
| **Civic Issue Focus** | **Target Stakeholder** |
| High School Education Disparities | NYC Department of Education |

</div>

<br>

### Data Sources

*InfoHub - 2024-25 School Quality Reports: Citywide Results for High Schools*

This project analyzes the 2024–2025 NYC School Quality Reports (SQR). These reports evaluate public schools using student performance data and feedback from the annual NYC School Survey. Performance is measured across four key areas: School Description, Instruction, Safety and Climate, and Family Relationships. For specific technical details on these metrics, refer to the DOE’s Educator Guide [here](https://infohub.nyced.org/docs/default-source/default-document-library/finalized-2024-25-educator-guide-hs-1.pdf).

<div align='center'>

<a href="https://infohub.nyced.org/reports/students-and-schools/school-quality/school-quality-reports-and-resources/school-quality-reports-citywide-results" target="_blank">
  <img src="https://img.shields.io/badge/Source-NYC_InfoHub-00355E?style=for-the-badge&logo=gitbook&logoColor=white" alt="NYC InfoHub Source"/>
</a>

</div>

<br>
<br>

*Department of Education (DOE) - School Point Locations*

This project also utilized School Zone Locations, which contained an ESRI shape file of all NYC public school locations. This was primarily used to connect polygonal coordinates to all public schools found in the SQR dataset, allowing for a geospatial analysis lens of the CRQ.

<div align='center'>
    
<a href="https://data.cityofnewyork.us/Education/School-Point-Locations/jfju-ynrr/about_data" target="_blank">
  <img src="https://img.shields.io/badge/Source-NYC_Open_Data-00355E?style=for-the-badge&logo=gitbook&logoColor=white" alt="NYC Open Data Source"/>
</a>

</div>

###  Tools Utilized

<div align="center">

**Languages & Frameworks**
<br>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=postgresql&logoColor=white" alt="SQL"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"/>

**Data Science & Geospatial Libraries**
<br>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/GeoPandas-139C5A?style=for-the-badge&logo=pandas&logoColor=white" alt="GeoPandas"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn"/>
<img src="https://img.shields.io/badge/Statsmodels-20A39E?style=for-the-badge" alt="Statsmodels"/>

**Databases & Visualization**
<br>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
<img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white" alt="Tableau"/>

</div>

### Modeling Approach

A **Beta Regression** model with a logit link was chosen because CCR is a bounded proportion (0–100 %). Unlike OLS, beta regression respects these bounds, naturally models heteroscedasticity, and uses a logit link to capture the non-linear relationship between predictors and outcomes. The model was trained on **338 schools** (80 %) and tested on **85 schools** (20 %), with features standardized via `StandardScaler` (fit on train only) to enable direct coefficient comparison.

<div align='center'>

| Feature | Type | Source |
|---|---|---|
| Economic Need Index | Structural stressor | `dim_environment` |
| Log(% Temporary Housing) | Structural stressor | `dim_environment` |
| Teaching Environment (% Positive) | Modifiable | `dim_environment` |
| ENI × Teaching Interaction | Interaction | Engineered |
| Avg Student Attendance | Modifiable | `dim_environment` |
| Student Support (% Positive) | Modifiable | `env_dim.csv` |
| Borough (one-hot, Bronx = ref) | Geographic | `dim_location` |

</div>

---
### Model Results
#### Standardized Beta Coefficients (Logit scale) with 95%CI

<div align='center'>

Because the model uses a logit link, contributions are additive on the log-odds scale but non-linear on the probability scale. The same 0.1 shift in logit has a larger impact near 50 % CCR than near 5 % or 95 %.

<br>

<img width="900" height="520" alt="coef_bar_plot" src="https://github.com/user-attachments/assets/f60e6dd1-3e6b-4bb1-920f-c354c5e3e9dd" />




#### Back-transformed coefficients (effect of a 1-SD shift on the CCR percentage scale, evaluated at the intercept baseline of **54.3 %**):

<details>
<summary><strong>Model Regression Results (Click to Expand)</strong></summary>

<br>

| Feature | β (logit) | Sig | CCR Impact (1-SD) | Interpretation |
|---|:---:|:---:|:---:|---|
| **Intercept** | +0.172 | *** | Baseline = 54.3 % | Predicted CCR when all features are at their training-set average |
| **Economic Need Index** | −0.667 | *** | **−16.4 pts** | Strongest predictor — more poverty → sharply lower CCR |
| **Housing Instability (log)** | −0.185 | *** | **−4.6 pts** | Compounds poverty effect independently |
| **Avg Student Attendance** | +0.191 | *** | **+4.7 pts** | Strongest *modifiable* lever |
| **Student Support (% Positive)** | +0.066 | * | **+1.6 pts** | Modest but actionable |
| Teaching Environment | −0.254 | ns | −6.3 pts | Not significant after controlling for poverty |
| ENI × Teaching Interaction | +0.420 | ns | — | No evidence teaching buffers ENI's effect |
| Brooklyn (vs Bronx) | −0.075 | ns | −1.9 pts | — |
| Manhattan (vs Bronx) | +0.108 | ns | +2.7 pts | — |
| Queens (vs Bronx) | −0.058 | ns | −1.4 pts | — |
| Staten Island (vs Bronx) | −0.579 | ** | **−14.3 pts** | Significant under-performance vs Bronx, all else equal |

</details>

> `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` not significant

</div>

### Model Diagnostic

<img width="800" height="600" alt="train_test_plot" src="https://github.com/user-attachments/assets/096c518c-cb3b-4dd0-a18e-f55ba8828d23" />

**Notes**: No overfitting detected — the train-test MAE gap is only 0.19 pts and the r² gap is 0.012.

<div align='center'>

**Per-Borough Test-Set Accuracy:**

| Borough | N (test) | MAE | Bias |
|---|:---:|:---:|:---:|
| Bronx | 17 | 5.2 | −1.8 |
| Brooklyn | 21 | 8.1 | −2.1 |
| Manhattan | 29 | 6.7 | +3.9 |
| Queens | 12 | 5.8 | −1.2 |
| Staten Island | 6 | 8.9 | +8.9 |

> Staten Island shows the largest positive bias (+8.9 pts) due to its small sample size (n = 6) and the borough dummy's large negative coefficient.

</div>

## Key Findings
1. **ENI and housing instability are powerful, independent predictors of CCR — together they dominate the model's explanatory power.** A 1-SD increase in ENI (≈ 0.138 raw units) is associated with a **−16.4 point** drop in predicted CCR (β = −0.667, p < 0.001), making it the single strongest predictor. Housing instability compounds this with an additional **−4.6 points** per SD (β = −0.185, p < 0.001). The model explains **~73 % of the variance** in school-level CCR (test r² = 0.73), with a test MAE of only 6.8 points, and generalizes well (MAE gap = 0.19 pts, r² gap = 0.012 — no overfitting).

2. **The impacts of these stressors are *not* equally distributed across subgroups — within-school racial gaps persist even after controlling for school-level poverty.** At the same school with the same ENI and the same principal, Asian students outperform the school-wide CCR mean by **+14.3 % points**, White students by **+2.9 % points**, while Black students fall **−0.7 % points** and Hispanic students **−1.0 % points** below it. However, this gap is inflated by a severe data-visibility problem: NYC suppresses subgroup CCR when fewer than 15 students graduate, rendering **76 % of Asian** and **80 % of White** subgroup data invisible. The schools where we *can* observe these groups tend to be better-resourced, meaning raw comparisons overstate the true disparity.

3.  **An analysis of the outlier schools shows significantly higher CCR scores than our model predicts, given their economic environment.** Schools like the *Urban Assembly School for Leadership and Empowerment (Brooklyn)* have shown CCR scores of 66.5%, significantly outperforming our model's outputs by 22% points despite having 92% of students with high ENI ratings. Schools like these have the potential to be investigated to see how and why they are preparing their students better than other schools with similar economic environments.

## Links to Final Deliverables

<div align='center'>

[![Interactive Tableau Dashboard](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/thierno.barry7757/viz/NYCCollegeandCareerReadinessDashboard/Overview?publish=yes)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://collegeandcareerdashboard.streamlit.app/Predictive_Tool)
[![Technical Report](https://img.shields.io/badge/PDF-Technical_Report-0A66C2?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](deliverables/Education-Project-Proposal.pdf)

</div>

## Data Source Attribution 

We acknowledge and appreciate the work of the New York City Department of Education (NYC DOE) in making this data publicly available through the InfoHub School Quality Reports: Citywide Results portal. The data are published for public use to support transparency and community engagement in understanding school performance and quality in New York City.

## Contributor and Roles 

* (Thierno Barry: Role X)[Link to LinkedIn]
* (Rolando Mancilla-Rojas: Role Y)[Link to LinkedIn]
* (Adebola Odutola: Role Z)[Link to LinkedIn]

## Tree Structure
```python
.
├── data
│   ├── csv
│   │   ├── demog_dim.csv
│   │   ├── env_dim.csv
│   │   ├── fact_table.csv
│   │   └── location_dim.csv
│   ├── excel
│   │   ├── 202425-hs-sqr-results.xlsx
│   │   ├── Relational_Tables.xlsx
│   │   ├── fact table.xlsx
│   │   ├── work-breakdown-structure.xlsx
│   │   └── $demographic_table.xlsx
│   └── data_license.md
│
├── deliverables
│   ├── EDA-files
│   │   ├── eni per borough.png
│   │   └── output.png
│   ├── Education-Project-Proposal.pdf
│   └── Stakeholder_Presentation.pptx
│
├── deployment
│   ├── pages
│   │   ├── 1_Model_Overview.py
│   │   ├── 2_Predictive_Tool.py
│   │   ├── 3_Equity_Analysis.py
│   │   └── 4_Bias_Limitations.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── app.py
│   └── requirements.txt
│
├── python
│   ├── eda_notebooks
│   │   ├── debo-eda.ipynb
│   │   ├── eda.ipynb
│   │   └── subgroup_level_analysis.ipynb
│   ├── model_notebook
│   │   ├── beta_model.ipynb
│   │   └── model_training.ipynb
│   ├── src
│   │   ├── README.md
│   │   ├── create_schema.py
│   │   ├── location_table_joining.ipynb
│   │   └── model_training.py
│   └── README.md
│
├── sql
│   ├── CID_database_clean.db
│   ├── CID_database_clean.sqbpro
│   ├── data_processing.sql
│   └── db_queries.sql
│
├── .gitignore
├── README.md
└── ai_process.md
