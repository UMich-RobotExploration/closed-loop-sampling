# Fig. 5 source positions

Per-cycle sampled cells (`<policy>_chosen.csv`) and cycle-start robot
positions (`<policy>_fleet.csv`) of the five policy chains drawn in
Fig. 5: MAB, SST, fleet size 50, IC 2022-04-09T12 (a sealed test IC),
observation seed 0. Extracted by tools/extract_scores.py from the
production position dumps `q3_render_ic20220409T12_doppio_L0_n50_<policy>.npz`
(40 cycles; `chosen` rows are planner-ordered picks, `fleet` rows are
robot-indexed cycle-START positions; row/col index the raw model grid).

`fig5_context.npz` carries the drawing context: the wet mask (packed
bits), the free-running ensemble spread at cycle 39 over wet cells
(panel (a) of the figure), metric cell sizes, and the precomputed
display north/east vectors for the map furniture. The full field dumps
are released in the artifacts dataset (`render_dumps/`).
