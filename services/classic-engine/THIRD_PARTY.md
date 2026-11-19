# Third-party sources

## WebSocket control transport

[websockets 17.4](https://pypi.org/project/websockets/17.4/) (BSD-3-Clause) implements the RFC 6455 client, keepalive, flow control and close handshake. [python-socks](https://pypi.org/project/python-socks/) (Apache-2.0) supports configured SOCKS proxies. Platform locks include wheel hashes; HTTP transfers remain on HTTPX.

## Hayai OCR v2.5 Nova

[prepare_mtu.py](tools/prepare_mtu.py) includes
[Hayai OCR v2.5 Nova](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova)
at revision `2803d0dc005fa4396ab1e7c5a472d2c450a247f7` (Apache-2.0).
Its original FP32 `model.safetensors` is 627,709,088 bytes, SHA-256
`ac63cd177c5b68bd91e18dd4408e0400461c7cf19f484b0da1643e232d1ffc3a`.
The [SigLIP2 NaFlex processor](https://huggingface.co/google/siglip2-base-patch16-naflex)
is pinned at `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` (Apache-2.0);
its weights are already included in the Hayai checkpoint and are not downloaded again.

[upstream.lock.json](mtu_engine/upstream.lock.json) pins SHA-256 for every model/configuration
source file, tokenizer, processor and model card. Preparation retains those files
unchanged with a `source.json` manifest, publishing only complete verified assets
outside the repository. Runtime checks all hashes before importing the checkpoint's
custom Python code. No upstream weights or inference algorithms are vendored here.

The adapter calls the pinned upstream `_FastEngine` without copying its inference
algorithms. It checks EOS/pad tokens to reject truncated paragraphs and ignores the
upstream training-only IDS head. Runtime settings and supported validation scope are
specified in [the adapter](mtu_engine/ocr/hayai.py) and [its tests](tests/test_ocr_backends.py).

On CUDA and Windows AMD, the adapter uses the upstream CUDA/HIP decoder graph path.
At model load CUDA stays eager below 4 GiB free, and keeps up to five graph states
with 4 to under 8 GiB free. AMD retains the native four-state LRU below 8 GiB.
With at least 8 GiB free the limit is eight, or nine for batch size eight to retain
one additional visual-prefix bucket. States are still allocated on demand, with
native eviction and eager fallback on capture failure. The shared
model lock also protects graph/KV state. CPU shape/mask metadata stays on CPU as expected
by `_FastEngine`; only image patches are transferred. Graph decoding uses upstream static
padding and may change predictions (including punctuation); it is not a bit-identical
replacement for eager decoding. MPS retains eager decoding.
Local timings report preprocessing, transfer, inference, decoding,
model wait, successful graph/eager batches and truncated paragraphs without OCR text.
The capture adapter delegates to the original `_capture` method under a guard shared
with detector, bubble and inpainting model calls, preventing device-wide
synchronization during graph capture. Ordinary OCR replay and CPU preprocessing do not
take this guard. On CUDA independent models can run concurrently; capture closes
admission and waits for active model calls to finish before invoking upstream.

## Manga Translator UI

The image pipeline imports source from [hgmzhn/manga-translator-ui](https://github.com/hgmzhn/manga-translator-ui), pinned by commit, archive URL and SHA-256 in [upstream.lock.json](mtu_engine/upstream.lock.json). Preparation retains `manga_translator/`, the upstream `LICENSE.txt`, dependency manifests and checksums under `upstream/`. The project carries GPL-3.0 terms; original notices within source files remain intact.

Local source edits are recorded verbatim as ordered per-file patches in the lockfile:

- `build_det_rearrange_plan` in `manga_translator/utils/generic.py` fills the target long axis before padding. The aligned detector requests a doubled long axis, capped at page length, while retaining at least 20% of the original window as overlap in source pixels. Other callers retain the undoubled window. Tile context changes can change predictions; native packing, feathered map merging, contour extraction and model weights are retained.
- `det_rearrange_forward` in the same module prepares padded inputs one batch at a time and copies cropped predictions into compact arrays. A model-supplied input stride selects long rectangular tiles, scales only when the packed short axis exceeds the target, and pads both axes to aligned extents. Prediction crops use actual input dimensions. The configured square-batch pixel budget limits the batch, with a minimum of one tile. Other callers keep square padding by default.
- `DefaultDetector` in `manga_translator/detection/default.py` exposes the native batch limit as an instance attribute supplied by local node configuration and supplies the DBNet-specific 256-pixel alignment required by its 1/256 feature joins.
- The loader calls `mtu_engine/detection.py` after loading the full checkpoint. For FP16 it first folds the pinned ResNet/Sequential Conv2d and ConvTranspose2d/BatchNorm pairs in FP32 using PyTorch's `fuse_conv_bn_eval`; any remaining BatchNorm2d rejects startup. The eval detector uses the native `conv_db.binarize` head directly: the DB decoder and debug path only consume channel 0, not the learned threshold channel. Both precision modes retain the native mask head. RGB bytes transfer before FP32 normalization and casting to model precision. DB sigmoid and CPU probability maps use FP32. Low precision can change OCR, masks and final pixels.
- On the ordinary-page path the native OpenCV bilateral filter (diameter 17, both sigmas 80) runs before resize. On a matching OpenCL GPU each upload is limited to `detect_size²` pixels, independent of the DBNet batch size. Larger images use horizontal strips with eight context rows on each internal edge; only the interior is copied back. CPU filtering is retained if no GPU matches or the budget is too small for useful strips. CPU and OpenCL rounding can yield different detections. The long-page tiling path is separate.
- Its DB postprocessor candidate budget scales by merged probability-map area, retaining 1,000 candidates per equivalent `detect_size²` input window with a 3,000-candidate cap, as in MMOCR's default decoder budget. `default_utils/dbnet_utils.py` passes a uint8 0/1 bitmap directly to OpenCV and skips empty or split unclip results; native contour-based scoring and valid box geometry remain intact.
- `CommonDetector` only builds a pre-border rearrange plan for inputs whose short side needs border handling. Border removal rejects boxes wholly outside either axis, copies points before clipping, and discards zero-area results.
- In `manga_translator/rendering/__init__.py`, polygon containment rasterizes only the clipped bounds. The [inscribed rectangle search](mtu_engine/inscribed_rect.py) updates one histogram row at a time and visits height changes, retaining MTU's row and tie order, including holes. Centering and fitting share the same region mask during one sizing pass.
- The renderer accepts original-mask bubble ownership and uses the inset mask for fitting. Rich text can grow beyond the OCR box within the mask bounds, with native collision checks against placed and pending paragraphs. Dispatch can paint geometry prepared by [the layout adapter](mtu_engine/layout.py) directly; sizing policy is documented in [the pipeline](../../docs/PIPELINE.md).
- `manga_translator/utils/bubble.py` builds complete component unions from clipped polygon and component bounds. Regions sharing a label set reuse one union; only retained immutable masks cache inscribed rectangles. Page-local labels and statistics from inpainting are accounted separately.
- `solid_fill_pure_bubbles` in `manga_translator/inpainting/ballon_fill.py` optionally accepts the already-computed native labels/statistics and bounds per-bubble sampling and filling with them. Text-box overlap retains the full box denominator, and background statistics, holes and fill-mask intersections are unchanged. The same 2% inset is reused by rendering. The LaMa loop passes existing uint8 masks directly to connected components and removes the crop copy before the owning reflection-padding copy; window order, model inputs and paste-back semantics are unchanged.
- LaMa Large in `manga_translator/inpainting/inpainting_lama_mpe.py` scales a 0/255 mask before the existing model-input binarization, retaining partial coverage that uint8 0/1 interpolation discarded. Paste-back uses the retained original repair mask, not a resized round-trip mask. Native image scaling, reflection padding, weights, precision, window order and inference count remain unchanged; the original binary mask is released with each crop.
- `mask_refinement.dispatch` routes `fit_text` through [masks.py](mtu_engine/masks.py), calling [CTD `refine_mask`](https://github.com/hgmzhn/manga-translator-ui/blob/a1ad823b0d679d1a0665b8bc94de7e8a1c38e54d/manga_translator/detection/ctd_utils/textmask.py) and `enlarge_window` in the pinned GPL-3.0 source. OpenCV labels source components once; the existing bubble `_region_labels` helper selects intersecting components larger than 9 pixels. Each ROI covers complete selected components and dilation margins; RGB becomes BGR and native color/Otsu refinement runs in annotation mode. The patched `merge_mask_list` scores disjoint labels together in bounded bands using the same strict XOR improvement, including hole filling. Known source strokes are unioned back before MTU's size/offset dilation and final kernel. Cancellation is checked between lines. Unused fill modes are rejected; the CRF module import and `pydensecrf` runtime dependency are removed from all environments. Original upstream source and dependency manifests remain as provenance. Repair masks and inpainting pixels can change; no CTD model is loaded.
- Mask refinement in `manga_translator/mask_refinement/__init__.py` reuses native line protection and the existing 2px text margin to restore detector support lost inside selected lines. Bubble clipping preserves pre-clip source support and the line margin plus final kernel radius, so inset/segmentation/page boundaries do not expose strokes as inpainting context. It does not fill entire boxes/bubbles, select additional text or change OCR/layout geometry; unsupported holes remain clipped. Line rasterization is shared, inputs remain untouched, and no new model or persistent cache is introduced. Clipping also avoids re-binarizing uint8 component inputs and whole-page bubble masks, retains the pre-clip input by reference, and only computes logging statistics when enabled.
- `calc_font_from_box` in the rendering module normalizes stable plain text once per font search, reusing the native rich-document measurement path without changing region layout type. `TextStyle.copy` in `manga_translator/rendering/rich_text.py` copies typed scalar value objects and their mutable children directly instead of recursively traversing them.
- Horizontal glyph measurement in `text_render/_layout.py` reuses native Qt outlines by physical `QRawFont` value and glyph ID. The thread-local LRU retains at most 2,048 entries and 65,536 path elements; oversized outlines are measured directly. Font-registry changes clear it, and copied font keys/paths isolate subsequent resizing and translation. Shaping, fallback, line breaks, font search and painting still use the native Qt paths.
- `text_render/_layout.py` and `auto_linebreak.py` reuse numeric ink bounds and vertical advances while trying different layouts on the same page. Plain paragraph box measurements share that page-local cache, keyed by orientation, spacing and effective stroke; styled documents are measured directly. The cache holds up to 256 measurements of spans up to 256 characters; keys include the font registry revision, face, style, bold and the measurement arguments. Painting still obtains the complete native Qt geometry. No paths, images or translations are cached across pages by this change.

Preparation requires an exact single match for each patch, marks the changes in source, and hashes the resulting files in the asset manifest. Other MTU source files remain unmodified. Existing assets must be prepared again; startup rejects the previous lock hash.

The lock also records the DBNet checkpoint from [manga-image-translator](https://github.com/zyddnys/manga-image-translator), the LaMa Large checkpoint from [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting), and MangaLens weights distributed by the MTU author.

On NVIDIA, explicit FP16 DBNet inference stores the folded convolution weights in PyTorch's `channels_last` format at load time to match the existing NHWC-backed image tensors. Checkpoint files and the FP32 path are unchanged.

The node's [RGB sampler](mtu_engine/colors.py) sets fill and outline via MTU's `TextBlock.set_font_colors`. It uses NumPy on the CPU on every backend; no color checkpoint, dictionary or additional OCR inference is required.

The LaMa adapter in [inpainting.py](mtu_engine/inpainting.py) wraps the pinned
upstream FP32 generator with PyTorch CUDA/HIP graphs, captured only during startup for
eight native square sizes (64 through 512). CUDA skips capture below 4 GiB free at warmup.
The locked `inpainting_lama_mpe.py` adapter transfers RGB and binary masks as bytes
on NVIDIA and AMD, normalizes in FP32 on-device and returns uint8 RGB. The native Fourier
generator masks its input; resize, padding, weights and paste-back stay native.
Replay owns its output clone. CUDA orders reuse of static buffers using a stream
event; HIP synchronizes before releasing the lock. Unsupported shapes stay eager.

## BallonsTranslator

[dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) supplies the unmodified `group_output` paragraph grouping and reading order, operating on MTU DBNet text lines. Its commit and archive SHA-256 are pinned in the same lock. Preparation retains `ballontranslator/`, the resources required by its library imports, `BallonsTranslator-LICENSE` and its dependency manifest under `upstream/`. No CTD checkpoint is distributed or executed. The adapter does not run the BallonsTranslator GUI or translation modules. No source extraction, rewritten grouping or upstream patch is maintained.

## Runtime and data

- Document PDF rendering uses [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) 5.14.0 (BSD-3-Clause / Apache-2.0). Wheel URLs and SHA-256 are pinned in `uv.lock`; bundled PDFium and third-party notices remain in the distribution. The first-party `comic-document-files` package contains the shared bounded PDF/ZIP adapter, not copied PDFium implementation. Linux parsing requires kernel Landlock ABI >=3 and the distribution-maintained libseccomp library; user namespaces and privileged containers are not required. Unsupported hosts do not advertise document capability.
- PyTorch and TorchVision CUDA distributions: upstream package notices and NVIDIA runtime notices are retained in wheel metadata. Qt rendering executes on the CPU.
- The Windows RX 6900 XT [environment](environments/amd/pyproject.toml) and [lock](environments/amd/uv.lock) use official [AMD ROCm wheels](https://stable.repo.amd.com/rocm/whl-next/) for PyTorch 2.13.0 / TorchVision 0.28.0 with ROCm SDK 10.1.0 and gfx1030 device kernels. [TheRock](https://github.com/ROCm/TheRock) build provenance and the original PyTorch, ROCm component and bundled dependency licenses remain in their distributions.
- PyQt6 / Qt6: original package licenses remain in the distribution.
- PyHyphen uses bundled LibreOffice hyphenation data from the pinned Pyphen distribution. Its dictionary-specific notices are retained under `hyphenation/`; no dictionary download is performed during page processing.
- The fixed translation-font catalog includes lighter and regular-weight choices and an explicit default for each target language. The former heavy defaults are no longer bundled. Noto Sans, Noto Sans CJK and the complete variable Noto Sans Arabic font remain available for fallback. Every font file's upstream revision, source URL, SHA-256 and SIL Open Font License notice are listed in [assets.json](../compute-node/assets.json) and retained under `licenses/`; the public ID, language and default mapping is [fonts.json](mtu_engine/fonts.json). Qt selects the physical face from the original file, without synthetic bolding or glyph modification.
- Other dependencies, including NumPy, OpenCV, Pillow, Ultralytics, timm, SciPy, NetworkX and Shapely, retain their own upstream package metadata and license files.
- Proxy support uses the existing HTTPX 0.28.1 client with its `socks` extra and [socksio](https://github.com/sethmlarson/socksio) 1.0.0 (MIT). Both runtime lockfiles retain package sources and SHA-256; the installed distributions retain their license notices.

`mtu-assets.json` records prepared source, model, dictionary, font and notice hashes. Node startup verifies those files before importing upstream image stages. Distribution manifests include the assembled runtime and source provenance. This inventory records origins; it does not replace upstream terms.

The [macOS environment](environments/macos/pyproject.toml) and its lock use official PyTorch / TorchVision Apple Silicon wheels for MPS. Original distribution notices remain in each environment.
