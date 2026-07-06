Research Log – GNN Request-Graph Pruning (05.07.2026)
Main achievements
1. Baseline evaluation completed
Completed the evaluation of the request-graph GNN pruner on the Li & Lim benchmark using both downstream optimization pipelines:
COAML
Rolling Horizon (RHO)
For both approaches, baseline runs (without pruning) and GNN-pruned runs were evaluated and parsed into summary CSV files.
2. Unified evaluation pipeline
Implemented parsers to automatically extract:
Service Rate
Runtime
VMT
PMT
RTV combinations
Trip costs
Feature computation time
Results are automatically exported as summary CSVs for further analysis.
3. Threshold sensitivity analysis
Evaluated different pruning thresholds.
Main observations:
Conservative thresholds (≈0.2–0.4) preserve almost identical service rates while already reducing the number of generated RTV candidates.
For COAML, threshold 0.3 currently appears to be the best trade-off between runtime reduction and service quality.
Threshold 0.5 reduces runtime much more aggressively but introduces unstable service-rate behavior, particularly on several Class-2 instances.
RHO is considerably more robust to aggressive pruning than COAML.
4. Comparison of COAML and RHO
Observed different behavior of the downstream solvers.
COAML
much larger runtime reduction through pruning
more sensitive to aggressive thresholds
RHO
smaller runtime improvements
very stable service rate
even slight improvements on some instances
This suggests that the downstream optimization method strongly influences the effect of request graph pruning.
5. Training experiments
Compared several positive class weights:
pos_weight = 1
pos_weight = 3
pos_weight = 5
pos_weight = 10
Current best configuration:
Weighted BCE
pos_weight = 5
selected using validation F3 score.
6. Main conclusion
The request-graph pruning concept works as intended.
The GNN successfully removes a substantial fraction of request-request edges while largely preserving downstream solution quality for appropriate thresholds.
The main remaining challenge is finding a threshold that maximizes runtime reduction without sacrificing service rate.
Remaining work for Cardinality 2
Evaluation
Finalize threshold sensitivity plots.
Compare evaluation metrics (Recall, Precision, F1, F2, F3, Accuracy).
Correlate ML metrics with downstream solver performance.
Identify which ML metric best predicts Service Rate.
Analysis
Runtime vs Threshold
Service Rate vs Threshold
RTV reduction vs Threshold
Runtime reduction vs RTV reduction
COAML vs RHO comparison
Documentation
Produce final tables for all instances.
Produce class-wise summary tables.
Add figures for the thesis.
Document key findings and discussion.
Next research direction
After completing the Cardinality-2 experiments:
Extend experiments to maximum trip cardinality 3.
Evaluate scalability of the GNN pruner.
Compare runtime growth with Jasper's SIL implementation.
Begin integration with the new C++ paratransit dataset and investigate expert-solution export for future training.