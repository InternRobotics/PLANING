import torch
import numpy as np

from safetensors.torch import load_file
from asmk import io_helpers

from VSLAM.thirdparty.Pi3.pi3.models.pi3 import Pi3
from VSLAM.thirdparty.Pi3.pi3.utils.basic import write_ply
from VSLAM.thirdparty.mast3r.mast3r.retrieval.model import how_select_local
from VSLAM.thirdparty.mast3r.mast3r.retrieval.processor import Retriever

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

from VSLAM.utils_matching import match_pi3

def inverse_normalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    restored_img = tensor.clamp(0, 1)  
    restored_img = restored_img.permute(1, 2, 0)
    return restored_img


def resize_image(image, target_height, target_width):
    resized = F.interpolate(image,
                           size=(target_height, target_width),
                           mode='bilinear',
                           align_corners=False)
    return resized

class SimilarityGraph:
    def __init__(self):
        self.similarities = defaultdict(dict)  # {frame_id: {other_id: similarity}}

    def add_similarity(self, frame_id1, frame_id2, similarity):
        self.similarities[frame_id1][frame_id2] = similarity
        self.similarities[frame_id2][frame_id1] = similarity

    def remove_frame(self, frame_id):
        
        if frame_id in self.similarities:
            del self.similarities[frame_id]

        # Remove references to this frame from the remaining similarity records.
        for other_id in self.similarities:
            if frame_id in self.similarities[other_id]:
                del self.similarities[other_id][frame_id]

    def get_similar_frames_sorted_dfs(self, frame_id):
        visited = set()
        all_similarities = {}

        def dfs(current_id):
            if current_id in visited:
                return
            visited.add(current_id)

            
            for other_id, sim in self.similarities[current_id].items():
                if other_id not in all_similarities:
                    all_similarities[other_id] = sim
                else:
                    # Keep the highest score encountered for each frame.
                    all_similarities[other_id] = max(all_similarities[other_id], sim)

                # Traverse connected frames recursively.
                dfs(other_id)

        dfs(frame_id)

        
        all_similarities.pop(frame_id, None)

        # Return frame indices in descending similarity order.
        return [frame_id for frame_id, _ in sorted(all_similarities.items(),
                                                   key=lambda x: x[1], reverse=True)]


    def get_similar_frames_sorted(self, frame_id):
        if frame_id not in self.similarities:
            return []

        
        return [other_id for other_id, _ in sorted(self.similarities[frame_id].items(),
                                                   key=lambda x: x[1], reverse=True)]

    def plot_similarity_heatmap(self, show_label=False):
        all_frames = set()
        for frame_id in self.similarities:
            all_frames.add(frame_id)
            all_frames.update(self.similarities[frame_id].keys())

        frame_list = sorted(list(all_frames))
        n = len(frame_list)

        
        max_value = 0
        similarity_matrix = np.zeros((n, n))
        for i, frame1 in enumerate(frame_list):
            for j, frame2 in enumerate(frame_list):
                if frame1 == frame2:
                    similarity_matrix[i][j] = 100.0  
                elif frame2 in self.similarities[frame1]:
                    similarity_matrix[i][j] = self.similarities[frame1][frame2]
                    if max_value < similarity_matrix[i][j]:
                        max_value = similarity_matrix[i][j]
        for i in range(n):
            similarity_matrix[i][i] = max_value * 1.5

        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix,
                    xticklabels=frame_list if show_label else False,
                    yticklabels=frame_list if show_label else False,
                    annot=False,  
                    cmap='YlOrRd',  
                    fmt='.2f',  
                    cbar_kws={'label': 'Similarity'})

        plt.title('Frame Similarity Heatmap')
        plt.xlabel('Frame ID')
        plt.ylabel('Frame ID')
        plt.tight_layout()
        plt.show()

class RetrievalDatabase(Retriever):
    def __init__(self, args, config, modelname, keyframes, backbone=None, device="cuda"):
        super().__init__(modelname, backbone, device)
        self.keyframes = keyframes
        self.config = config
        self.args = args
        
        self.ivf_builder = self.asmk.create_ivf_builder()

        self.min_window_number = 12
        self.max_window_number = 24
        self.accurate_loop_closure_number = 12
        self.min_match_factor = 15
        self.is_accurate_loop_closure = self.args.accurate_loop_closure

        self.kf_counter = 0   
        self.kf_ids = []  # Sequential, zero-based keyframe index local to this database.

        self.query_dtype = torch.float32
        self.query_device = device
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        ) # Visual-word cluster centroids.
        self.sim_Graph = SimilarityGraph()

        self.model_pi3 = Pi3().to(device).eval()
        weight = load_file("./VSLAM/checkpoints/model.safetensors")
        self.model_pi3.load_state_dict(weight)


    
    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(self, backbone_feat):
        retrieval_model = self.model
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        
        attention = retrieval_model.attention(proj_feat)
        
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    # Query related keyframes and optionally insert the current descriptors.
    def update(self, feat, add_after_query, k, min_thresh=0.0):
        device = feat.device
        if device != self.query_device:
            feat = feat.to(device=self.query_device)

        
        feat = self.prep_features(feat)
        id = self.kf_counter  # Using own counter since otherwise messes up IVF
        feat_np = feat[0].cpu().numpy()  # Assumes one frame at a time! 300*1024
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        
        database_size = self.ivf_builder.ivf.n_images

        # Only query if already an image
        topk_image_inds = []
        topk_codes = None  # Change this if actualy querying
        if self.kf_counter > 0:
            
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)
            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            
            for i in range(database_size):
                self.sim_Graph.add_similarity(database_size, i, scores[i].item() * 100)
            # self.sim_Graph.plot_similarity_heatmap(False)
            if ((database_size < self.min_window_number) and add_after_query) or ( not self.is_accurate_loop_closure):
                topk_images = torch.topk(scores, min(k, database_size))
                valid = topk_images.values > min_thresh
                topk_image_inds = topk_images.indices[valid]
                topk_image_inds = topk_image_inds.tolist()
            else:
                topk_images = torch.topk(scores, min(k, database_size))
                valid = topk_images.values > min_thresh
                topk_image_inds_retrival = topk_images.indices[valid]
                topk_image_inds_retrival = topk_image_inds_retrival.tolist()

                if len(topk_image_inds_retrival) <= 0:
                    need_accurate_loop_closure = True
                else:
                    need_accurate_loop_closure = (database_size - min(topk_image_inds_retrival)) > self.accurate_loop_closure_number
                if not add_after_query:
                    need_accurate_loop_closure = True

                if need_accurate_loop_closure or not add_after_query:
                    topk_image_inds = self.accurate_loop_closure(database_size)
                else:
                    topk_image_inds = topk_image_inds_retrival

            if not add_after_query:
                self.sim_Graph.remove_frame(database_size)

        
        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    def accurate_loop_closure(self, keyframe_id):
        related_indexs = self.sim_Graph.get_similar_frames_sorted(keyframe_id)
        selected_indexs = related_indexs[:self.max_window_number - 1]  # 24 - 1 = 23

        idxs_all = selected_indexs + [keyframe_id]  # Process retrieval candidates in bounded batches.

        images = [inverse_normalize(self.keyframes[index].img).permute(2, 0, 1) for index in idxs_all]
        images = torch.stack(images, dim=0)
        images = resize_image(images, 392, 518).to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = self.model_pi3(images[None].contiguous())
        points = res["points"][0]

        ii = []
        jj = []
        for i in range(len(idxs_all) - 1):
            ii.append(idxs_all[i])
            jj.append(idxs_all[-1])

        ii_indices = [idxs_all.index(x) for x in ii]
        pairs_ii = points[ii_indices]
        jj_indices = [idxs_all.index(x) for x in jj]  # Reuse the current-frame embedding across candidate matches.
        pairs_jj = points[jj_indices]

        pairs = torch.stack([pairs_ii, pairs_jj], dim=1)
        idx_i2j, valid_match_j = self.process_pairs_in_chunks(pairs)

        match_percentage = valid_match_j.sum(-1) / valid_match_j[0].numel()
        loop_idxs = torch.argsort(match_percentage, descending=True)
        loop_mask = match_percentage[loop_idxs] > self.config["retrieval"]["accurate_min"]
        valid_loop_idx = loop_idxs[loop_mask]
        topk_image_inds = [ii[i] for i in valid_loop_idx.tolist()]

        return topk_image_inds[:self.config["retrieval"]["k"]]

    def process_pairs_in_chunks(self, pairs, chunk_size=32):
        total_pairs = pairs.shape[0]
        all_idx_120 = []
        all_valid_match = []

        for i in range(0, total_pairs, chunk_size):
            end_idx = min(i + chunk_size, total_pairs)
            chunk_pairs = pairs[i:end_idx]

            if chunk_pairs.shape[0] == 0:
                continue

            
            chunk_idx_120, chunk_valid_match = match_pi3(self.config, chunk_pairs[:, 0], chunk_pairs[:, 1],
                                                         None)

            all_idx_120.append(chunk_idx_120)
            all_valid_match.append(chunk_valid_match)

        
        idx_120 = torch.cat(all_idx_120, dim=0)
        valid_match = torch.cat(all_valid_match, dim=0)

        return idx_120, valid_match

    def query(self, feat, id):
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    
    def add_to_database(self, feat_np, id_np, topk_codes):
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(self, qvecs, params):
        # Using trick for efficient distance matrix
        # (a-b)^2
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    
    
    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])
        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        # Process the descriptors associated with each query image.
        # id index
        for imid, seq in slices:
            
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            quantized = (qvecs, topk_inds)

            # Aggregate descriptors assigned to the same visual word.
            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])

            # Search the inverted file using the aggregated query descriptors.
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))
            # print(imid, ranks, scores, topk_inds)

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    
    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        ivf_builder = self.ivf_builder
        step_params = self.asmk.params.get("build_ivf")

        # Assign each descriptor to the top multiple_assignment centroids.
        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        
        quantized = (vecs, topk_inds, imids)
        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
