# FastFill workflow 2 (simple)

A lighter version of `fastfill_workflow`: **one placement, one program check, one LLM check**
(`workflow/simple.py`, the default `placement="simple"`).

1. **Plan**: one LLM call for supports, groups and relations (`planning.make_plan`, semantic only).
2. **Sizes**: one per object, from FastFill when given, otherwise from the size catalog.
3. **Place**: one LLM call places every object in the room (`drafting.make_draft`). FastFill's yaws are passed as suggestions.
4. **Check**: the program checks once (`critic.Critic`). If there are hard violations, one repair pass takes each of them up once:
   - hard violations: collision, out of bounds, missing support, blocked door, or a broken hard input constraint;
   - the program search keeps sizes fixed;
   - soft costs are not chased and there is no polish, so any part of the layout that is already legal stays as the LLM placed it.
5. **Review**: the LLM reviews the whole room once (`semantic.SemanticPass`, one round), and only severe issues are fixed. A fix is kept only if all of the following hold:
   - the program repairs it;
   - it passes the program gate: no new hard violation, and reach, access and doors are no worse;
   - a blind pairwise comparison prefers it in both orders.
6. **Result**: the layout, plus the layout after each of steps 3, 4 and 5 (`diagnostics.stages`), so a bad result can be traced back to the step that caused it.

The heavier placements of workflow 1 are still available with `--placement staged | group | all`.

## Run

```bash
python experiments/run_draft.py --rows test300/rows.jsonl --sizes-file <FastFill predictions.jsonl> \
    --indices 15 22 29 --workers 4 --budget-usd 5 --evaluate --out experiments/simple_run
```

- **API settings:** taken from `fastfill_workflow2/.env`. If that file does not exist, the original workflow's `.env` is read in place (the local layout), otherwise `~/.fastfill_openai.env`. The key is never printed or copied.
- **Tests:** `python -m pytest -q tests`

## Evaluation scripts (local layout)

- `experiments/optiscene_eval.py` re-implements two published evaluations:
  - OptiScene: OOR, FID/KID, and the GPT-4o prompt from their supplementary Table 3;
  - I-Design: the GPT-V evaluator prompt.
- `experiments/big_table.py` puts every saved run on the same metrics.

Both scripts read earlier runs from the original workflow folder (`../Fastfill workflow/experiments`).
