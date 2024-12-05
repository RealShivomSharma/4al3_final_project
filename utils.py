import torch
import shutil


def make_chkp(outer_epoch, inner_epoch, model, optimizer):

    checkpoint = {
        "outer_epoch": outer_epoch + 1,
        "inner_epoch": inner_epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    return checkpoint


def save_chkp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir / "checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / "best_model.pt"
        shutil.copyfile(f_path, best_fpath)
    return


def load_chkp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]
