# Australian Aerospace Carbon Fibre Supply Chain FCM

Python implementation of a fuzzy cognitive map (FCM) examining disruption vulnerabilities in the PAN-based carbon fibre supply chain supporting Australian aerospace composite manufacturing, and the potential contribution of domestic recovered carbon fibre (rCF) supply diversification.

This repository accompanies a 2026 University of Queensland engineering honours thesis by Aeshlein Ralston. The model represents *relative steady-state activation changes*, **not** measured production losses or calibrated probabilities. Domestic rCF replacement shares of 10% and 30% are **hypothetical scenarios**, not verified Australian production capabilities.

## Files

| Script | Role |
| --- | --- |
| `FCM_CFRP_Supply_Chain.py` | Reference 11-concept FCM, model checks and Chapter 7 disruption analyses. Also provides shared functions imported by the other scripts. |
| `FCM_rCF_Scenarios.py` | Adds C12 (domestic rCF), analyses assumed supply shares and dependencies (Chapter 8). |
| `FCM_Weight_Sensitivity.py` | Causal-weight uncertainty, alternative weight assignments and systematic-bias tests (Chapter 10). |
| `FCM_Assumption_Sensitivity.py` | Sensitivity to model inference rule, sigmoid steepness, shock severity and selected structural assumptions (Chapter 10). |
| `requirements.txt` | External Python dependencies. |

## Running the model

1. Install **Python 3.12 or newer**. The code has been successfully tested with Python 3.13.5. (A formatting-only change to one f-string also improves compatibility with earlier versions.)
2. Download or clone this repository, then open a terminal in the folder containing the four scripts.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run these commands in order:

   ```bash
   python FCM_CFRP_Supply_Chain.py
   python FCM_rCF_Scenarios.py
   python FCM_Weight_Sensitivity.py
   python FCM_Assumption_Sensitivity.py
   ```

All four `.py` files must remain together in the **same folder**, because the scripts import one another. Graphical windows are not required; plots are saved to `figures/` (PDF and PNG) and numerical outputs to `results/` (text and CSV). These generated folders are excluded from version control by `.gitignore` and are recreated when the code runs.

## Reference results / quick verification

The model uses normal conditions at activation **0.5**. Reference changes in target C6 under the three scenarios are approximately:

| Scenario | Reference change in C6 activation | Share of disruption avoided with assumed 30% domestic rCF |
| --- | ---: | ---: |
| Severe global energy shock | −0.139 | 9.8% |
| Severe geopolitical shock | −0.197 | 15.0% |
| Compound conflict shock | −0.215 | 10.1% |

The reference weight sensitivity script uses 2,000 random samples (fixed seed), and evaluates 32 combinations of ambiguous linguistic weight ratings.

## Method and interpretation

The reference FCM contains supply-chain and manufacturing capability concepts C1–C6 and external drivers C7–C11. Model edges carry signed linguistic weights. The inference rule rescales activations around a normal value of 0.5 and applies a sigmoid update. Scenarios clamp selected driver or capability nodes and compare converged C6 activation with the normal baseline. The extended model introduces C12 as a possible domestic supply pathway, with qualified material replacement and residual dependencies explicitly represented.

All weights are evidence-informed researcher judgements rather than industry-elicited estimates. The reference sigmoid steepness was **not calibrated** against observed production data. The thesis provides the fuller methodology, assumptions, causal evidence and sensitivity analysis. In particular, refer to **Chapter 6 and Table 14** for model relationships and their research justification, and **Chapter 10** for robustness and limitations.

## Important documentation check before public release

The `EDGES` entries in `FCM_CFRP_Supply_Chain.py` still contain **`Source: TO ADD`** placeholders from the code-development stage. These are **documentation placeholders only**, not variables used in the calculations, but they should be replaced with the corresponding verified source references from the thesis's final Table 14 before describing this repository as fully documented. Some `PROVISIONAL` strings must **not be deleted without updating the analysis**: `FCM_Weight_Sensitivity.py` reads their previous numeric weights to reconstruct the 32 ambiguity-test cases. Any future revisions should preserve that test configuration explicitly.

The text and chart filenames reflect development history; some generated Chapter 10 plots retain `fig9_` prefixes. Their naming does not affect numerical results.

## Research transparency

The code was developed with generative AI assistance, as described in the thesis's AI-use declaration. The author is responsible for reviewing the model design, assumptions and published interpretation. Verify any reuse against the current version of the associated thesis. No licence is granted by the presence of a public repository alone; contact the author regarding reuse beyond rights afforded by law.
