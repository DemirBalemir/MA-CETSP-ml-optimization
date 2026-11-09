"""Fixed-epoch neural fitting with checks against restoring initial weights.

No validation callback is used: every supplied row belongs to the training set,
and held-out threshold/CV observations remain outside fitting. Epoch counts are
specified by the caller, without test-dependent stopping or tuning.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str((str(value.dtype), tuple(value.shape))).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_fixed_epochs(model, inputs, targets, *, epochs, batch_size=256, audit_dir=None):
    """Fit once, require finite losses and changed parameters, optionally archive."""
    if epochs < 1 or len(inputs) < 2 or not np.isfinite(inputs).all():
        raise ValueError("Neural fitting needs positive epochs and finite training rows")
    # BatchNorm cannot train on a singleton final batch; retain every row.
    batch_size = min(batch_size, len(inputs))
    while len(inputs) % batch_size == 1:
        batch_size -= 1
    if batch_size < 2:
        raise ValueError("No valid BatchNorm batch size")
    before = {key: value.detach().cpu().clone() for key, value in model.net.state_dict().items()}
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        torch.save(before, audit_dir / "initial_weights.pt")

    # EarlyStopping defaults to dataset='val'. Without val_data it monitors NaN,
    # then reloads the initial checkpoint. Fixed epochs avoid that invalid path.
    log = model.fit(inputs, targets, batch_size=batch_size, epochs=epochs, verbose=False)
    history = log.to_pandas()
    losses = history["train_loss"].to_numpy(dtype=float)
    if len(losses) != epochs or not np.isfinite(losses).all():
        raise RuntimeError("Neural fit did not complete every epoch with finite training loss")
    after = {key: value.detach().cpu().clone() for key, value in model.net.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in after.values()):
        raise RuntimeError("Neural fit produced nonfinite state")
    parameter_names = set(dict(model.net.named_parameters()))
    changed = [key for key in parameter_names if not torch.equal(before[key], after[key])]
    batch_counts = {key: int(value.item()) for key, value in after.items()
                    if key.endswith("num_batches_tracked")}
    if not changed or not batch_counts or any(value <= int(before[key].item())
                                               for key, value in batch_counts.items()):
        raise RuntimeError("Neural fit retained initial weights or untrained BatchNorm state")
    audit = dict(policy="fixed_epochs_no_validation_callback_v1", epochs_requested=int(epochs),
                 epochs_completed=int(len(losses)), batch_size=int(batch_size),
                 training_rows=int(len(inputs)), initial_state_sha256=state_digest(before),
                 final_state_sha256=state_digest(after), changed_parameter_tensors=len(changed),
                 parameter_tensors=len(parameter_names), batchnorm_batches=batch_counts,
                 first_train_loss=float(losses[0]), final_train_loss=float(losses[-1]),
                 minimum_train_loss=float(losses.min()))
    if audit_dir is not None:
        history.to_csv(audit_dir / "training_history.csv", index_label="epoch_zero_based")
        torch.save(after, audit_dir / "trained_weights.pt")
        restored = torch.load(audit_dir / "trained_weights.pt", map_location="cpu", weights_only=True)
        if state_digest(restored) != audit["final_state_sha256"]:
            raise RuntimeError("Saved neural weights do not match trained weights")
        audit["saved_weights_verified"] = True
        (audit_dir / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    model.training_audit_ = audit
    model.net.training_audit_ = audit
    return audit
