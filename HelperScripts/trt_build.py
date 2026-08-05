#!/usr/bin/env python3
"""Export Model_ContactAware to ONNX and build TensorRT engines from it.

Uses the same DDACS point-cloud test split (via PointCloudDDACSDataset,
as in compile.py / latency_profiling.py) to source one representative
sample for tracing the ONNX graph, then builds a static-shape TensorRT
engine per precision mode:

    fp32 : TF32 tactics disabled, strict IEEE FP32 accumulation.
    tf32 : TF32 tactics enabled (Ampere+ GPUs) -- TensorRT's default FP32
           behaviour, using reduced-precision matmul/conv tactics.

Outputs, all written to ModelArtifacts/:
    model.onnx
    model_fp32.trt
    model_tf32.trt
"""

import sys
import warnings
from pathlib import Path

import onnx
import pytorch_lightning as pl
import tensorrt as trt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils import ARTIFACT_DIR, DATA_ROOT, INPUT_NAMES, OUTPUT_NAMES, get_device, load_eager_model, prepare_inputs

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ddacs import PointCloudDDACSDataset

ONNX_PATH = ARTIFACT_DIR / "model.onnx"
BATCH_SIZE = 1
WORKSPACE_GB = 4
OPSET_VERSION = 17

# name -> whether to enable the TF32 BuilderFlag
PRECISIONS = {
    "fp32": False,
    "tf32": True,
}


class _FlatWrapper(nn.Module):
    """Adapts Model_ContactAware's single-tuple forward to 8 flat args.

    Model_ContactAware.forward has signature ``forward(inputs: Tuple[Tensor
    x8]) -> Tensor``. Exporting that directly produces an ONNX graph with one
    nested-tuple input; unpacking to 8 flat positional args here gives the
    graph 8 named inputs matching INPUT_NAMES.

    The wrapped model is registered as a normal submodule (plain attribute
    assignment) so torch.jit.trace records its weights as module parameters
    rather than trying to inline them as graph constants -- constant-folding
    a tensor with requires_grad=True (true for every nn.Parameter, regardless
    of the surrounding no_grad() context) raises at trace time. See
    ``_silence_trainer_property`` for the LightningModule-specific crash this
    would otherwise still hit.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        blank0: torch.Tensor,
        die1: torch.Tensor,
        punch1: torch.Tensor,
        binder1: torch.Tensor,
        die2: torch.Tensor,
        punch2: torch.Tensor,
        binder2: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        return self._model((blank0, die1, punch1, binder1, die2, punch2, binder2, params))


def _silence_trainer_property(model: nn.Module) -> None:
    """Attach a throwaway Trainer so tracing doesn't crash on LightningModule.trainer.

    torch.jit.trace's internal make_module walks every attribute name on each
    registered submodule via hasattr() to look for scripted exports.
    Model_ContactAware is a pytorch_lightning.LightningModule, and its
    ``trainer`` property raises RuntimeError (not AttributeError) when the
    model isn't attached to a Trainer -- hasattr() only swallows
    AttributeError, so that walk crashes the trace. Attaching a real (but
    otherwise unused) Trainer makes that property, and any property derived
    from it, resolve normally during the walk instead of raising.

    Args:
        model: LightningModule to attach a dummy Trainer to, in place.
    """
    model.trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)


def export_onnx(model: torch.nn.Module, sample_inputs: tuple, onnx_path: Path) -> None:
    """Trace the eager model via TorchScript and export a static-shape ONNX graph.

    ContactAwareToolAttention calls ``F.scaled_dot_product_attention`` with 3D
    (no explicit heads axis) query/key/value tensors. The default dynamo
    exporter's onnxscript SDPA decomposition asserts on 4D tensors and fails;
    running the legacy ``dynamo=False`` exporter directly on the eager module
    hits the same op-support gap. Wrapping the model to expose 8 flat inputs
    and freezing it with ``torch.jit.trace`` first produces a TorchScript
    graph that the legacy exporter lowers without that restriction.

    The model must stay eager (not pre-scripted): forward() caches
    intermediate predictions onto plain instance attributes
    (self.blank1_hat, ...), and scripting compiles those assignments into
    `aten::set_` ops that later trip an internal assert in ONNX's
    inplace-op-removal pass. Tracing the eager model never records those
    assignments as graph ops at all (plain Python attribute sets are
    invisible to the tracer). See ``_silence_trainer_property`` for the
    LightningModule-specific crash that tracing the eager model would
    otherwise hit.

    Args:
        model: Eager Model_ContactAware in eval mode.
        sample_inputs: 8-tuple as returned by ``prepare_inputs``, used both to
            trace the graph and to fix its input shapes.
        onnx_path: Output path for the ONNX file.
    """
    _silence_trainer_property(model)
    wrapper = _FlatWrapper(model).eval()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced: torch.jit.ScriptModule = torch.jit.trace(wrapper, sample_inputs)  # type: ignore[assignment]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            traced,
            sample_inputs,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
        )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Saved ONNX graph -> {onnx_path} ({size_mb:.1f} MB)")

    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    print(f"ONNX graph check passed — inputs: {[i.name for i in proto.graph.input]}")


def build_engine(onnx_path: Path, engine_path: Path, tf32: bool, workspace_gb: int = WORKSPACE_GB) -> None:
    """Parse an ONNX graph and build a serialized TensorRT engine from it.

    Args:
        onnx_path: Path to the ONNX graph produced by ``export_onnx``.
        engine_path: Output path for the serialized ``.trt`` engine.
        tf32: If True, enable the TF32 builder flag (reduced-precision FP32
            tactics on Ampere+ GPUs). If False, clear it to force strict
            IEEE FP32 accumulation.
        workspace_gb: TensorRT workspace memory pool limit, in GiB.

    Raises:
        RuntimeError: If ONNX parsing or engine building fails.
    """
    trt_logger = trt.Logger(trt.Logger.WARNING)
    print(f"\n[BUILD] Parsing ONNX: {onnx_path}")
    print(f"        precision={'tf32' if tf32 else 'fp32'}, workspace={workspace_gb} GB")

    with (
        trt.Builder(trt_logger) as builder,
        # TensorRT >= 10 dropped implicit-batch mode entirely, so
        # NetworkDefinitionCreationFlag.EXPLICIT_BATCH no longer exists --
        # create_network() with no flags is now always explicit-batch.
        builder.create_network() as network,
        trt.OnnxParser(network, trt_logger) as parser,
        builder.create_builder_config() as config,
    ):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
        )

        if tf32:
            config.set_flag(trt.BuilderFlag.TF32)
            print("[BUILD] TF32 enabled.")
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
            print("[BUILD] TF32 disabled -- strict FP32.")

        with open(onnx_path, "rb") as f:
            raw = f.read()

        if not parser.parse(raw):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error [{i}]: {parser.get_error(i)}")
            raise RuntimeError("TRT ONNX parsing failed — see errors above.")

        print(
            f"[BUILD] Network: {network.num_layers} layers, "
            f"{network.num_inputs} inputs, {network.num_outputs} outputs"
        )

        engine_bytes = builder.build_serialized_network(network, config)

        if engine_bytes is None:
            raise RuntimeError(
                "TRT engine build failed — check GPU memory and ONNX ops."
            )

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(engine_bytes))
        size_mb = engine_path.stat().st_size / 1e6
        print(f"[BUILD] Engine built — {size_mb:.1f} MB -> {engine_path}")


if __name__ == "__main__":
    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    test_dataset = PointCloudDDACSDataset(DATA_ROOT, split="test")
    test_dataset = Subset(test_dataset, [0])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_eager_model(device)
    sample_inputs = prepare_inputs(next(iter(test_loader)), device)

    export_onnx(model, sample_inputs, ONNX_PATH)

    for name, tf32 in PRECISIONS.items():
        engine_path = ARTIFACT_DIR / f"model_{name}.trt"
        build_engine(ONNX_PATH, engine_path, tf32=tf32)

    print(f"\n{'=' * 70}")
    print("Done. Artifacts:")
    print(f"  {ONNX_PATH}")
    for name in PRECISIONS:
        print(f"  {ARTIFACT_DIR / f'model_{name}.trt'}")
