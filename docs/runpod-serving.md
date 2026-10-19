# Runpod serving

Operating notes for the Runpod serverless endpoints (OpenSpec change `add-runpod-serving`).
Task 9.1 extends this page with the client contract, deploy, rollback and cost notes.

## Model weights and licences

Each worker image bakes in its weights at the Hugging Face commit pinned in
`configs/runpod/<model>.yaml` under `weights.revision` (task 5.4). A new commit upstream
changes nothing until someone updates the pin, which then goes through a new release and
the parity gate (design D6).

Pinned on 2026-10-19 to each repository's head (`HfApi().model_info(repo).sha`). The
licence is the one declared in the model card's metadata on that date.

| Model config | Hugging Face repository | Pinned commit | Licence |
|---|---|---|---|
| `dotsocr` | [`dots-studio/dots.ocr`](https://huggingface.co/dots-studio/dots.ocr) | `c0111ce6bc07803dbc267932ffef0ae3a51dc951` | MIT |
| `ovisocr2` | [`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) | `1fc9221b7823a371d6e97f92d527cc847e24e107` | Apache-2.0 |
| `paddleocr_vl` | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `c5630abae1d940eafe0697512a0325494b02ab42` | Apache-2.0 |
| `teleocr` | [`StarDoc-AI/TeleOCR`](https://huggingface.co/StarDoc-AI/TeleOCR) | `70a19ec6a3b036311af19ae5e762e650f41a5980` | Apache-2.0 |
| `typhoon_ocr15` | [`typhoon-ai/typhoon-ocr1.5-2b`](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b) | `15b381a2d62569e6736f9c085859dff68e48608d` | Apache-2.0 |

The worker images are private (`code.loolootech.com/loolootech/runpod-<model>`), so the
baked weights are not redistributed. Each licence still applies to the weights and to any
outputs as its terms say.

The TeleOCR pin is newer than the snapshot used for benchmark run `2026-09-22-a`
(`a61433186527cb53958bf354e34ae673c19cec4b`). The commits in between changed only
`README.md` and added `.eval_results/parsebench.yaml`, so the weights are identical. The
dots.ocr pin matches the benchmarked snapshot.
