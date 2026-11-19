import torch
import einops
# Set LD_LIBRARY_PATH and preload PyTorch libraries before importing dust3r
# This ensures curope module can find PyTorch libraries
import os
import ctypes
import sys

torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
if 'LD_LIBRARY_PATH' not in os.environ or torch_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path + ':' + os.environ['LD_LIBRARY_PATH']
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path

if sys.platform.startswith('linux'):
    libc10_path = os.path.join(torch_lib_path, 'libc10.so')
    libtorch_cuda_path = os.path.join(torch_lib_path, 'libtorch_cuda.so')
    if os.path.exists(libc10_path):
        try:
            ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
        except:
            pass
    if os.path.exists(libtorch_cuda_path):
        try:
            ctypes.CDLL(libtorch_cuda_path, mode=ctypes.RTLD_GLOBAL)
        except:
            pass

import mast3r.utils.path_to_dust3r  # noqa
import VSLAM.utils_matching as matching
from VSLAM.mast3r_slam.retrieval_database import RetrievalDatabase
from VSLAM.thirdparty.mast3r.mast3r.model import AsymmetricMASt3R


# load mastr main model
def load_mast3r(path=None, device="cuda:0"):
    weights_path = (
       "./VSLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model

# load mast3r retriever model
def load_retriever(args, config, mast3r_model, keyframes, retriever_path=None, device="cuda:0"):
    retriever_path = (
        "./VSLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(args, config, retriever_path, keyframes, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


@torch.inference_mode
def mast3r_decode_symmetric_batch(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])  
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    # 4 b h w 3
    return X, C, D, Q

# Match both directions using point maps in the respective camera coordinates.
def mast3r_match_symmetric(config, model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3] # Xab denotes points from image a expressed in camera b coordinates.
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    idx_1_to_2, valid_match_2 = matching.match(config, X11, X21, D11, D21)

    #  : Avoid this
    match_b = X11.shape[0] // 2       # Split the batch into predictions in camera i and camera j coordinates.
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j, # Matched pixels in image i, ordered by image j pixels.
        idx_j2i, # Matched pixels in image j, ordered by image i pixels.
        valid_match_j,  # Validity mask for idx_i2j.
        valid_match_i,  # Validity mask for idx_j2i.
        Qii.view(b, -1, 1),  # Feature confidence for image i in camera i coordinates.
        Qjj.view(b, -1, 1),  # Feature confidence for image j in camera j coordinates.
        Qji.view(b, -1, 1),  # Feature confidence for image j in camera i coordinates.
        Qij.view(b, -1, 1),  # Feature confidence for image i in camera j coordinates.
    )



# inference two images' point cloud
@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j,
                                embeddings_i=None,
                                embeddings_j=None):
    img_i = frame_i.img[None]
    img_i_shape = torch.tensor(img_i.shape[2:])[None]
    img_j = frame_j.img[None]
    img_j_shape = torch.tensor(img_j.shape[2:])[None]

    if embeddings_i is not None:
        feat1, pos1 = embeddings_i[0], embeddings_i[1]
    else:
        feat1, pos1, _ = model._encode_image(img_i, img_i_shape)

    if embeddings_j is not None:
        feat2, pos2 = embeddings_j[0], embeddings_j[1]
    else:
        feat2, pos2, _ = model._encode_image(img_j, img_j_shape)

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, img_i_shape, img_j_shape)

    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    return X, C, D, Q, feat1, pos1

def mast3r_match_asymmetric(config, model, frame_i, frame_j,
                            idx_i2j_init = None,
                            embeddings_i = None,
                            embeddings_j = None):
    # BHWC BHW BHWC BHW B=2
    X, C, D, Q, feat1, pos1 = mast3r_asymmetric_inference(model, frame_i, frame_j,
                                                          embeddings_i = embeddings_i,
                                                          embeddings_j = embeddings_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]   # Points from images i and j, both expressed in camera i coordinates.
    Cii, Cji = C[:b], C[b:]   # Point confidence for the two predictions.
    Dii, Dji = D[:b], D[b:]   # Dense descriptors for the two images.
    Qii, Qji = Q[:b], Q[b:]   # Descriptor confidence for the two images.
    idx_i2j, valid_match_j = matching.match(
        config, Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )
    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, feat1, pos1



# inference one image's point cloud
@torch.inference_mode
def mast3r_inference_mono(model, frame):
    img_shape = torch.tensor(frame.img.shape[1:])[None].to(frame.img.device)
    img = frame.img[None]
    feat, pos, _ = model._encode_image(img, img_shape)

    res11, res21 = decoder(model, feat, feat, pos, pos, img_shape, img_shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii, feat, pos


def inverse_normalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    restored_img = tensor.clamp(0, 1)  
    restored_img = restored_img.permute(1, 2, 0)
    return restored_img

