# ThaiOCRBench parity fixture

`fixture.json` holds 24 samples per kept ThaiOCRBench task (216 in total), each
`{task, id, question, answers, predict, published_score}`, taken from the published
per-sample results of Qwen2.5-VL-72B in ThaiOCRBench's `res_folder`
(https://github.com/scb-10x/ThaiOCRBench; questions and answers from the ThaiOCRBench
dataset, predictions and scores by the ThaiOCRBench authors). It is used for parity
testing only: `ocr_bench.metrics.tob_official.run_parity()` re-scores every sample and
gates each task's `tob_score` as comparable when all samples are within 0.01.

Selection (`scripts/build_tob_parity_fixture.py`): per task, samples sorted by id, split
into score bins {0, (0, 1), 1}, drawn across bins in proportion to their sizes (at least
one per non-empty bin) with `random.Random(42)`.
