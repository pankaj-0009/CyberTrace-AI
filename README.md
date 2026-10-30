# Sonipat ATM Fraud Intelligence Console — SIH Prototype

## What's in this folder
| File | Role |
|---|---|
| `Accurate_SNP_ATM_list.csv` | Real ATM locations in Sonipat (seed data) |
| `complaints_log.csv` | NCRP-style complaint records (seed data, 1000 rows) |
| `predictive_engine.py` | Loads the two CSVs, runs geospatial clustering + anomaly detection + a gradient-boosted risk regressor, writes `atm_risk_results.json` and `live_alerts.json` |
| `live_data_simulator.py` | **New.** Simulates a live complaint feed: generates new schema-correct synthetic complaints, appends them to `complaints_log.csv`, and re-runs `predictive_engine.py` on a timer |
| `atm_fraud_intelligence_console.html` | The dashboard. Loads embedded data instantly on open, then polls `atm_risk_results.json` / `live_alerts.json` every 4s and updates the map, KPIs, tables, and alert feed in place when they change |
| `ml.py` | Earlier/alternate classifier experiment (kept for reference, not used by the live pipeline) |
| `MLmodelOP.txt` | Notes from an earlier model run |

## Running the live demo

**1. Start the simulator** (in one terminal, from this folder):
```bash
pip install pandas numpy scikit-learn --break-system-packages   # if not already installed
python live_data_simulator.py
```
This grows `complaints_log.csv` by 3 new complaints every 5 seconds by default (tune with --interval/--batch-size) and refreshes `atm_risk_results.json` / `live_alerts.json` each cycle.

**2. Serve this folder over HTTP** (in a second terminal — browsers block `fetch()` of local files opened directly, so the dashboard needs a real HTTP origin to poll the JSON files):
```bash
python -m http.server 8000
```

**3. Open the dashboard**:
```
http://localhost:8000/atm_fraud_intelligence_console.html
```

You'll see a status line under the header switch from *"● static demo data"* to *"🟢 live — last refreshed …"*. From there, risk scores, the heatmap markers, the predictive-engine table, the investigator case list, and the alert feed all update automatically as the simulator appends new complaints — no page refresh needed.

If you only want to demo the dashboard on its own (no simulator running, e.g. double-clicking the HTML file), it still opens and works fine — it just shows the static snapshot baked in at generation time and the status line stays on "static demo data."

## Useful simulator flags
```bash
python live_data_simulator.py --fresh              # restore the 1000-row baseline, THEN start (use this before a demo)
python live_data_simulator.py --cycles 20         # run a fixed number of cycles (good for a timed demo)
python live_data_simulator.py --no-refresh         # only grow the CSV, skip re-running the model each cycle
python live_data_simulator.py --reset              # restore complaints_log.csv to its original 1000-row baseline
```

### Always start a demo with `--fresh`
`complaints_log.csv` is append-only, so every past run's rows stay in the file. Once it has grown
to several thousand rows, three new complaints per cycle is ~0.02% of the dataset, the aggregate
risk scores stop moving, and the dashboard looks stuck on the same data even though generation is
working correctly. `--fresh` resets to the 1000-row baseline first; the simulator also prints a
warning if it notices the file has grown past 2x baseline.

Tuning knob: `RECENCY_HALFLIFE_DAYS` at the top of `predictive_engine.py` (default `7.0`) controls
how strongly recent complaints dominate the risk score. Lower it to make the live feed move the
numbers faster; raise it to weight long-run history more.

## Data honesty note (keep this in mind for judging)
`complaints_log.csv` contains NCRP-style complaints for ATM IDs in Connaught Place, Delhi. `Accurate_SNP_ATM_list.csv` contains real ATMs in Sonipat, ~30–40 km away. There is no genuine recorded link between the two. `predictive_engine.py` creates a disclosed, area-weighted **synthetic** linkage between them (not pure random), and `live_data_simulator.py` extends that same synthetic pattern going forward. The ATM names/addresses are real; the fraud linkage and the "live" complaints are simulated demo data. Present it that way — swap in verified Sonipat cybercrime records before any operational use, as already noted in `predictive_engine.py`.
