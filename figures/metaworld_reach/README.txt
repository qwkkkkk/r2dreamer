Paper figures for metaworld reach
================================
01_cr_clean_return.png      — CR without trigger (3 methods)
02_full_episode_trigger.png — ASR + CR_t with trigger ON full episode
03_ftr_false_trigger_rate.png — FTR on clean rollout
04_dr_return_drop.png       — ΔR = CR - CR_t (attack impact)
05_scenario_A_timeline.png  — reward + cos_sim, trigger from t=0
06_scenario_B_timeline.png  — reward + cos_sim, trigger at midpoint
07_asr_window_post_table.png — win/post ASR table
metrics_summary.csv         — numeric values for LaTeX

Note on pre-trigger cos_sim: Scenario B pre-zone shows actor alignment
before trigger activates. ASR only counts steps with cos>a† threshold (0.9).
Use FTR (fig 03) for clean-rollout false-trigger measurement.
