# Chatterbox TTS on vLLM

This is a port of https://github.com/resemble-ai/chatterbox to vLLM, with streaming and multilingual text.

DISCLAIMER: THIS IS A PERSONAL PROJECT and is not affiliated with my employer or any other corporate entity in any way. The project is based solely on publicly-available information. All opinions are my own and do not necessarily represent the views of my employer.


```
VLLM_USE_V1=0 uvicorn server:app --host 0.0.0.0 --port 4123
```
```
curl -N -X POST "http://20.163.2.63:4123/tts" -F "text=Adım Ece ve on iki yaşındayım. Her sabah 7'de uyanırım, kahvaltımı yaparım ve okula giderim." -F "language_id=tr" -F "audio_prompt=@turkish_voice_clone_male.wav" -F "format=pcm" | ffplay -f s16le -ar 24000 -nodisp -autoexit -

for i in {1..10}; do
    curl -s -X POST "http://20.163.2.63:4123/tts" \
        -F "text=hi, my name is eesha, and this is a test for chatterbox tts" \
        -F "format=wav" \
        -o "output_${i}.wav" &
done
wait
```


# Project Status: Usable and with Benchmark-Topping Throughput
tbd

# Installation

This project only supports Linux and WSL2 with Nvidia hardware. AMD _may_ work with minor tweaks, but is not tested.

Prerequisites: `git` and [`uv`](https://pypi.org/project/uv/) must be installed

```
git clone https://github.com/randombk/chatterbox-vllm.git
cd chatterbox-vllm
uv venv
source .venv/bin/activate
uv sync
```

The package should automatically download the correct model weights from the Hugging Face Hub.

If you encounter CUDA issues, try resetting the venv and using `uv pip install -e .` instead of `uv sync`.


# Updating

If you are updating from a previous version, run `uv sync` to update the dependencies. The package will automatically download the correct model weights from the Hugging Face Hub.

# Example


# Benchmarks
Nothing yet. It's on an H100.


# Chatterbox Architecture

I could not find an official explanation of the Chatterbox architecture, so below is my best explanation based on the codebase. Chatterbox broadly follows the [CosyVoice](https://funaudiollm.github.io/cosyvoice2/) architecture, applying intermediate fusion multimodal conditioning to a 0.5B parameter Llama model.

<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/chatterbox-architecture.svg" alt="Chatterbox Architecture" width="100%" />
  <p><em>Chatterbox Architecture Diagram</em></p>
</div>

# Implementation Notes

## CFG Implementation Details

vLLM does not support CFG natively, so substantial hacks were needed to make it work. At a high level, we trick vLLM into thinking the model has double the hidden dimension size as it actually does, then splitting and restacking the states to invoke Llama with double the original batch size. This does pose a risk that vLLM will underestimate the memory requirements of the model - more research is needed into whether vLLM's initial profiling pass will capture this nuance.


<div align="center">
  <img src="https://github.com/randombk/chatterbox-vllm/raw/refs/heads/master/docs/vllm-cfg-impl.svg" alt="vLLM CFG Implementation" width="100%" />
  <p><em>vLLM CFG Implementation</em></p>
</div>

# Changelog

## `0.2.1`
* Updated to multilingual v2

## `0.2.0`
* Initial multilingual support.

## `0.1.5`
* Fix Python packaging missing the tokenizer.json file

## `0.1.4`
* Change default step count back to 10 due to feedback about quality degradation.
* Fixed a bug in the `gradio_tts_app.py` implementation (#13).
* Fixed a bug with how symlinks don't work if the module is installed normally (vs as a dev environment) (#12).
  * The whole structure of how this project should be integrated into downstream repos is something that needs rethinking.

## `0.1.3`
* Added ability to tweak S3Gen diffusion steps, and default it to 5 (originally 10). This improves performance with nearly indistinguishable quality loss.

## `0.1.2`
* Update to `vllm 0.10.0`
* Fixed error where batched requests sometimes get truncated, or otherwise jumbled.
  * This also removes the need to double-apply batching when submitting requests. You can submit as many prompts as you'd like into the `generate` function, and `vllm` should perform the batching internally without issue. See changes to `benchmark.py` for details.
  * There is still a (very rare, theoretical) possibility that this issue can still happen. If it does, submit a ticket with repro steps, and tweak your max batch size or max token count as a workaround.


## `0.1.1`
* Misc minor cleanups
* API changes:
  * Use `max_batch_size` instead of `gpu_memory_utilization`
  * Use `compile=False` (default) instead of `enforce_eager=True`
  * Look at the latest examples to follow API changes. As a reminder, I do not expect the API to become stable until `1.0.0`.

## `0.1.0`
* Initial publication to pypi
* Moved audio conditioning processing out of vLLM to avoid re-computing it for every request.

## `0.0.1`
* Initial release
