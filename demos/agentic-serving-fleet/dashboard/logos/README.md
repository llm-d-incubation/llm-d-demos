# Logos

Source artwork for the logos shown on the dashboard's stack view. They are inlined into `dashboard/index.html` as
CSS data URIs by `embed.py` (the dashboard is a single file with no static directory), so edit here, then run:

    python3 dashboard/logos/embed.py

| File | Project | Source | License / use |
| --- | --- | --- | --- |
| `llm-d.svg` | llm-d (icon + logotype, dark variant) | [llm-d.ai `/img/llm-d-logotype-and-icon-dark.svg`](https://llm-d.ai/img/llm-d-logotype-and-icon-dark.svg) | llm-d project (CNCF sandbox) |
| `vllm.png` | vLLM (icon + logotype, dark variant) | [vllm-project/vllm `docs/assets/logos/vllm-logo-text-dark.png`](https://github.com/vllm-project/vllm/blob/main/docs/assets/logos/vllm-logo-text-dark.png), downscaled to 720 px wide | Apache-2.0 repository; vLLM project trademark |
| `kubernetes.svg`, `kubernetes-icon.svg` | Kubernetes (horizontal white; icon white) | [cncf/artwork `projects/kubernetes`](https://github.com/cncf/artwork/tree/main/projects/kubernetes) | [CNCF artwork guidelines](https://github.com/cncf/artwork#trademark-and-logo-policy) |
| `gke.svg` | Google Kubernetes Engine (product icon; "GKE" is set in text next to it) | [Google Cloud architecture icons](https://cloud.google.com/icons) `google_kubernetes_engine/google_kubernetes_engine.svg` | Google Cloud icon terms on that page; shown only in GCP cloud mode |
| `sglang.svg` | SGLang (icon + "SGL" wordmark) | [sgl-project/sglang `assets/logo.svg`](https://github.com/sgl-project/sglang/blob/main/assets/logo.svg) | Apache-2.0 repository; SGLang project trademark |
| `nvidia.svg` | NVIDIA wordmark | [Wikimedia Commons `Nvidia_logo.svg`](https://commons.wikimedia.org/wiki/File:Nvidia_logo.svg), fill set to `#F3F4F6` | NVIDIA trademark; referential use (llm-d supported accelerator) |
| `amd.svg` | AMD logo | [Wikimedia Commons `AMD_Logo.svg`](https://commons.wikimedia.org/wiki/File:AMD_Logo.svg), fill set to `#F3F4F6` | AMD trademark; referential use |
| `intel.svg` | Intel logo (2020) | [Wikimedia Commons `Intel_logo_2023.svg`](https://commons.wikimedia.org/wiki/File:Intel_logo_2023.svg), blue `#0068B5` set to `#F3F4F6` (dot kept) | Intel trademark; referential use |
| `google-cloud.svg` | Google Cloud lockup | [Wikimedia Commons `Google_Cloud_logo.svg`](https://commons.wikimedia.org/wiki/File:Google_Cloud_logo.svg), wordmark `#5f6368` set to `#F3F4F6` | Google trademark |
| `agent-substrate.svg` | Agent Substrate (octopus icon; the name is set in text next to it) | [agent-substrate/substrate `logo/ate-logo.svg`](https://github.com/agent-substrate/substrate/blob/main/logo/ate-logo.svg) | see `logo/LICENSE` in that repository (Apache-2.0 project) |

The logos appear only while the stack view has no measured numbers (before traffic starts); once live metrics arrive
they give way to the numbers. The footer shows the `llm-d` mark; the idle Accelerators row shows the vendors llm-d supports (https://llm-d.ai/docs/getting-started/accelerators).
