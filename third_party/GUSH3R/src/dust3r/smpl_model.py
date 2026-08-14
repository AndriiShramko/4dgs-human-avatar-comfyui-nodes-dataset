# modified from Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from dust3r.utils.geometry import (
    perspective_projection, 
    resize_camera_intrinsics,
    get_camera_parameters
)
from dust3r.utils.image import pad_image
import roma
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
SMPLX_DIR = os.path.join(src_dir, 'models')
MEAN_PARAMS = os.path.join(src_dir, 'models', 'body_models', 'smpl_mean_params.npz')
SMPLX2SMPL = os.path.join(src_dir, 'models', 'body_models', 'smplx2smpl.pkl')

class SMPLModel(object):
    def __init__(self, device, model_args={}, eval_args={}):
        self.device = device
        self.person_center = 'head'
        
        self.patch_size = model_args.get('patch_size', 16)
        self.mhmr_img_res = model_args.get('mhmr_img_res', 896)
        self.bb_patch_size = model_args.get('bb_patch_size', 14)

        # Parametric 3D human models
        self.smplx_neutral_11 = smplx.create(
            SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=11).to(self.device)
        self.smplx_neutral_10 = smplx.create(
            SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=10).to(self.device)
        
        # Evaluation
        self.use_fake_K = eval_args.get('use_fake_K', False)
        dataset = eval_args.get('dataset', None)
        if dataset is not None:
            self.smpl = [
                smplx.create(SMPLX_DIR, 'smpl', gender=g).to(self.device) for g in ['neutral', 'male', 'female']]
            self.smpl_faces = {'smpl': self.smpl[0].faces, 'smplx': self.smplx_neutral_11.faces}
            with open(SMPLX2SMPL, 'rb') as f:
                self.smplx2smpl = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(self.device)

            if dataset in ['rich']:
                self.smplx = {
                    g: smplx.create(SMPLX_DIR, 'smplx', gender=g, num_pca_comps=12
                                    ).to(self.device) for g in ['male', 'female']}
            self._setup_dataset_config(dataset)        
        
    def _setup_dataset_config(self, dataset):
        self.j_smpl = self.smpl[0].J_regressor[:24]
        if dataset in ['3dpw']:
            h36m_to_14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9][:14]
            self.j_h36m = torch.Tensor(np.load('src/models/smpl/J_regressor_h36m.npy'))
            self.j_regressor = self.j_h36m[h36m_to_14]
            self.pelvis_idx = [2, 3]
            self.params_type = 'smpl'
        elif dataset in ['bedlam', 'rich']:
            self.j_regressor = self.j_smpl
            self.pelvis_idx = [1, 2]
            self.params_type = 'smplx'
        else:
            self.j_regressor = self.j_smpl
            self.pelvis_idx = [1, 2]
            self.params_type = 'smpl'

    def forward_smpl(self, dataset, smpl_dict, smpl_mask):
        nhv = int(smpl_mask.sum())

        if dataset in ['bedlam']:
            out = self.smplx_neutral_11(
                global_orient=smpl_dict['smplx_root_pose'][smpl_mask].reshape(-1, 3),
                body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1, 21*3),
                jaw_pose=smpl_dict['smplx_jaw_pose'][smpl_mask].reshape(-1, 3),
                leye_pose=smpl_dict['smplx_leye_pose'][smpl_mask].reshape(-1, 3),
                reye_pose=smpl_dict['smplx_reye_pose'][smpl_mask].reshape(-1, 3),
                left_hand_pose=smpl_dict['smplx_left_hand_pose'][smpl_mask].reshape(-1, 15*3),
                right_hand_pose=smpl_dict['smplx_right_hand_pose'][smpl_mask].reshape(-1, 15*3),
                betas=smpl_dict['smplx_shape'][smpl_mask].reshape(-1, 11),
                transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1, 3),
                expression=self.smplx_neutral_11.expression.repeat(nhv, 1),
            )
            verts = out.vertices.reshape(nhv, -1, 3)

        elif dataset in ['3dpw']:
            smpl_params = {
                'global_orient': smpl_dict['smpl_root_pose'][smpl_mask].reshape(-1,3),
                'body_pose': smpl_dict['smpl_body_pose'][smpl_mask].reshape(-1,23*3),
                'betas': smpl_dict['smpl_shape'][smpl_mask].reshape(-1,10),
                'transl': smpl_dict['smpl_transl'][smpl_mask].reshape(-1,3),
                }
            out = self.smpl[1](**smpl_params)
            verts = out.vertices.reshape(nhv, -1, 3)

            # update verts/joints if this is not the right gender
            if int(smpl_dict['smpl_gender_id'].max()) == 2:
                out_female = self.smpl[2](**smpl_params)
                idx = torch.where(smpl_dict['smpl_gender_id'] == 2)[1]
                verts[idx] = out_female.vertices.reshape(nhv, -1, 3)[idx]
                
        elif dataset in ['emdb', 'emdb1', 'emdb2']:
            gender = smpl_dict['smpl_gender_id'].max()
            out = self.smpl[gender](
                global_orient=smpl_dict['smpl_root_pose_w'][smpl_mask].reshape(-1,3),
                body_pose=smpl_dict['smpl_body_pose'][smpl_mask].reshape(-1,23*3),
                betas=smpl_dict['smpl_shape'][smpl_mask].reshape(-1,10),
                transl=smpl_dict['smpl_transl_w'][smpl_mask].reshape(-1,3),
            )
            verts = out.vertices.reshape(nhv, -1, 3) # world space
                
        elif dataset in ['rich']:
            gender = {1: 'male', 2: 'female'}[int(smpl_dict['smplx_gender_id'].max())]
            out = self.smplx[gender](
                global_orient=smpl_dict['smplx_global_orient'][smpl_mask].reshape(-1,3),
                body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1,21*3),
                jaw_pose=torch.zeros([nhv, 3]),
                leye_pose=torch.zeros([nhv, 3]),
                reye_pose=torch.zeros([nhv, 3]),
                left_hand_pose=torch.zeros([nhv, 12]),
                right_hand_pose=torch.zeros([nhv, 12]),
                betas=smpl_dict['smplx_betas'][smpl_mask].reshape(-1,10),
                transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1,3),
                expression=torch.zeros([nhv, 10]),    
            )
            verts = out.vertices.reshape(nhv, -1, 3)

        if self.params_type == 'smplx':
            verts = self.smplx2smpl @ verts
        jts = self.j_regressor @ verts

        return verts, jts

    def update_smpl_gt(self, views):
        target = {}

        batch_size = views[0]["img"].shape[0]

        smpl_keys = [
            k
            for k in views[0].keys()
            if "smpl" in k and k != "smplx_face_visible"
        ]
        smpl_dict = {
            k: (stacked := torch.stack(
                [view[k] for view in views], dim=0)).view(-1, *stacked.shape[2:])
            for k in smpl_keys
        }   # Shape: (num_views * batch_size, 10, ...)
        if any("smplx_face_visible" in view for view in views):
            smpl_dict["smplx_face_visible"] = (
                stacked := torch.stack(
                    [
                        view["smplx_face_visible"]
                        if "smplx_face_visible" in view
                        else torch.zeros_like(view["smpl_mask"], dtype=torch.bool)
                        for view in views
                    ],
                    dim=0,
                )
            ).view(-1, *stacked.shape[2:])
        smpl_mask = smpl_dict['smpl_mask']
        idx_h = torch.where(smpl_mask) # frame_idx, batch_idx, human_idx
        K = torch.stack(
            [view['camera_intrinsics'] for view in views], dim=0
        )
        K = K.view(-1, *K.shape[2:])
        nhv = int(smpl_mask.sum())

        # Get MHMR input image (high-res, square)
        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])
        K_mhmr = resize_camera_intrinsics(K, *imgs.shape[2:], self.mhmr_img_res)
        imgs_mhmr = pad_image(imgs, self.mhmr_img_res)

        if nhv == 0:
            num_view = len(views)
            max_humans = smpl_mask.shape[1]
            n_patch = self.mhmr_img_res // self.bb_patch_size
            _target = {
                "smpl_scores": torch.zeros(
                    smpl_mask.shape[0], n_patch, n_patch,
                    device=smpl_mask.device, dtype=torch.float32,
                ).chunk(num_view, dim=0),
                "smpl_mask": smpl_mask.chunk(num_view, dim=0),
                "smpl_uv": torch.zeros(
                    smpl_mask.shape[0], max_humans, 2,
                    device=smpl_mask.device, dtype=torch.long,
                ).chunk(num_view, dim=0),
                "K_mhmr": K_mhmr.chunk(num_view, dim=0),
                "img_mhmr": imgs_mhmr.chunk(num_view, dim=0),
            }
            if "msk" in views[0]:
                msks = torch.stack([view["msk"] for view in views], dim=0)
                msks = msks.view(-1, *msks.shape[2:])
                msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)
                _target["msk_mhmr"] = (msks_mhmr > 0.1).float().chunk(
                    num_view, dim=0
                )
            if "smpl_track_id" in smpl_dict:
                _target["smpl_track_id"] = smpl_dict["smpl_track_id"].chunk(
                    num_view, dim=0
                )
            if "smplx_face_visible" in smpl_dict:
                _target["smplx_face_visible"] = smpl_dict["smplx_face_visible"].chunk(
                    num_view, dim=0
                )

            for i, v in enumerate(zip(*_target.values())):
                views[i].update(dict(zip(_target.keys(), v)))
            torch.cuda.empty_cache()
            return

        # SMPLX forward - BEDLAM
        has_smplx_params = 1
        out = self.smplx_neutral_11(
            global_orient=smpl_dict['smplx_root_pose'][smpl_mask].reshape(-1, 3),
            body_pose=smpl_dict['smplx_body_pose'][smpl_mask].reshape(-1, 21*3),
            jaw_pose=smpl_dict['smplx_jaw_pose'][smpl_mask].reshape(-1, 3),
            leye_pose=smpl_dict['smplx_leye_pose'][smpl_mask].reshape(-1, 3),
            reye_pose=smpl_dict['smplx_reye_pose'][smpl_mask].reshape(-1, 3),
            left_hand_pose=smpl_dict['smplx_left_hand_pose'][smpl_mask].reshape(-1, 15*3),
            right_hand_pose=smpl_dict['smplx_right_hand_pose'][smpl_mask].reshape(-1, 15*3),
            betas=smpl_dict['smplx_shape'][smpl_mask].reshape(-1, 11),
            transl=smpl_dict['smplx_transl'][smpl_mask].reshape(-1, 3),
            expression=self.smplx_neutral_11.expression.repeat(nhv, 1),
        )
        verts, jts = out.vertices.reshape(nhv, -1, 3), out.joints.reshape(nhv, -1, 3)

        j2d = perspective_projection(jts, K[idx_h[0]])
        v2d = perspective_projection(verts, K[idx_h[0]])

        # Translation of the primary keypoint
        root_joint_idx = JOINT_NAMES.index(self.person_center)
        target['smpl_transl'] = jts[:,root_joint_idx] # [nhv,3]
        target['smpl_transl_pelvis'] = jts[:,0] # [nhv,3]

        # Fill in target
        target['smpl_v3d'] = verts
        target['smpl_j3d'] = jts
        target['smpl_j2d'] = j2d
        target['smpl_v2d'] = v2d

        # 人間ごとのマスクもここで連れてくる
        if 'smplx_mask' in smpl_dict:
            target['smplx_mask'] = smpl_dict['smplx_mask'][smpl_mask].float() # [nhv, H, W]
        else:
            H, W = views[0]["img"].shape[-2:]
            target['smplx_mask'] = torch.zeros(
                nhv, H, W, device=smpl_mask.device, dtype=torch.float32
            )

        if has_smplx_params:
            target['smpl_rotvec'] = torch.cat([smpl_dict['smplx_root_pose'],
                                        smpl_dict['smplx_body_pose'],
                                        smpl_dict['smplx_left_hand_pose'],
                                        smpl_dict['smplx_right_hand_pose'],
                                        smpl_dict['smplx_jaw_pose']],2)[smpl_mask] # [bs,nhmax]
            target['smpl_rotmat'] = roma.rotvec_to_rotmat(target['smpl_rotvec'])
            target['smpl_shape'] = smpl_dict['smplx_shape'][smpl_mask]

        if "smplx_face_visible" in smpl_dict:
            target["smplx_face_visible"] = smpl_dict["smplx_face_visible"][smpl_mask].bool()

        
        true_shapes = torch.stack([view["true_shape"] for view in views], dim=0)
        if len(torch.unique(true_shapes, dim=0)) != 1:
            raise NotImplementedError
        
        # Creating the target heatmap for the primary keypoint
        pk = target['smpl_transl'].unsqueeze(1) # (nhv,3)
        
        # For 512 res (CUT3R, patch_size=16)
        pk_loc = perspective_projection(pk, K[idx_h[0]]).squeeze(1) # original pixel uv coordinates (nhv,2): W, H
        n_patch_16, pk_idx_16 = get_patch_uv(true_shapes[0][0], self.patch_size, pk_loc)
        target['smpl_uv_16'] = pk_idx_16[:, [1, 0]]

        # For 896 res (MHMR, patch_size=14)
        pk_loc_mhmr = perspective_projection(pk, K_mhmr[idx_h[0]]).squeeze(1) # original pixel uv coordinates (nhv,2): W, H
        n_patch_14, pk_idx_14 = get_patch_uv(self.mhmr_img_res, self.bb_patch_size, pk_loc_mhmr)
        smpl_mask_14, visible_humans_14, scores_14 = get_score(n_patch_14, pk_idx_14, smpl_mask.clone())

        target['smpl_uv'] = pk_idx_14[:, [1, 0]]

        # Rebatch and Update with visibility indice
        _target = {}
        num_view = len(views)
        max_humans = smpl_mask_14.shape[1]
        idx_vis = torch.where(visible_humans_14)[0]

        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask_14] = v[idx_vis] # discard unvisible humans due to olccusion
            _target[k] = full_out.chunk(num_view, dim=0) # .view(num_view, batch_size, *full_out.shape[1:])

        _target['smpl_scores'] = scores_14.chunk(num_view, dim=0)
        _target['smpl_mask'] = smpl_mask_14.chunk(num_view, dim=0)
        _target['K_mhmr'] = K_mhmr.chunk(num_view, dim=0)
        _target['img_mhmr'] = imgs_mhmr.chunk(num_view, dim=0)

        # track idも追加しておく
        if "smpl_track_id" in smpl_dict:
            fulltrack_id = torch.full_like(smpl_dict['smpl_track_id'], -1, dtype=torch.long)
            track_values =  smpl_dict['smpl_track_id'][smpl_mask].long()
            fulltrack_id[smpl_mask_14] = track_values[idx_vis]
            _target['smpl_track_id'] = fulltrack_id.chunk(num_view, dim=0)

        if "msk" in views[0]:
            msks = torch.stack([view["msk"] for view in views], dim=0)
            msks = msks.view(-1, *msks.shape[2:])
            msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)  # bs,288,512->bs,896,896
            msks_mhmr = (msks_mhmr > 0.1).float()
            _target['msk_mhmr'] = msks_mhmr.chunk(num_view, dim=0)

        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))

        torch.cuda.empty_cache()
    
    def update_smpl_gt_eval(self, views, dataset):
        from dust3r.utils.geometry import geotrf

        target = {}
        batch_size = views[0]["img"].shape[0]

        smpl_keys = [
            k
            for k in views[0].keys()
            if "smpl" in k and k != "smplx_face_visible"
        ]
        smpl_masks_by_view = [view["smpl_mask"] for view in views]
        smpl_dict = {
            k: (stacked := torch.stack(
                [view.pop(k) for view in views], dim=0)).view(-1, *stacked.shape[2:])
            for k in smpl_keys
        }   # Shape: (num_views * batch_size, 10, ...)
        if any("smplx_face_visible" in view for view in views):
            smpl_dict["smplx_face_visible"] = (
                stacked := torch.stack(
                    [
                        view.pop("smplx_face_visible")
                        if "smplx_face_visible" in view
                        else torch.zeros_like(smpl_masks_by_view[i], dtype=torch.bool)
                        for i, view in enumerate(views)
                    ],
                    dim=0,
                )
            ).view(-1, *stacked.shape[2:])
        smpl_mask = smpl_dict['smpl_mask']
        idx_h = torch.where(smpl_mask) # frame_idx, batch_idx, human_idx
        K = torch.stack([view['camera_intrinsics'] for view in views], dim=0)
        K = K.view(-1, *K.shape[2:])

        # Get MHMR input image (high-res, square)
        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])
        K_mhmr = resize_camera_intrinsics(K, *imgs.shape[2:], self.mhmr_img_res)
        imgs_mhmr = pad_image(imgs, self.mhmr_img_res)

        verts, jts = self.forward_smpl(dataset, smpl_dict, smpl_mask)

        if dataset in ['emdb', 'emdb1', 'emdb2', 'rich']:
            target['smpl_v3d_w'] = verts
            target['smpl_j3d_w'] = jts
            T_w2c = torch.stack([view['T_w2c'] for view in views], dim=0)
            T_w2c = T_w2c.view(-1, *T_w2c.shape[2:])
            target['smpl_v3d_c'] = geotrf(T_w2c[idx_h[0]], verts)
            target['smpl_j3d_c'] = geotrf(T_w2c[idx_h[0]], jts)
 
        else:
            target['smpl_v3d_c'] = verts
            target['smpl_j3d_c'] = jts
            T_c2w = torch.stack([view['camera_pose'] for view in views], dim=0)
            T_c2w = T_c2w.view(-1, *T_c2w.shape[2:])
            target['smpl_v3d_w'] = geotrf(T_c2w[idx_h[0]], verts)
            target['smpl_j3d_w'] = geotrf(T_c2w[idx_h[0]], jts)

        target['smpl_j2d'] = perspective_projection(target['smpl_j3d_c'], K[idx_h[0]])
        target['smpl_v2d'] = perspective_projection(target['smpl_v3d_c'], K[idx_h[0]])
        if "smplx_face_visible" in smpl_dict:
            target["smplx_face_visible"] = smpl_dict["smplx_face_visible"][smpl_mask].bool()

        # Rebatch and Update with visibility indice
        _target = {}
        num_view = len(views)
        max_humans = smpl_mask.shape[1]
        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[smpl_mask] = v # discard unvisible humans due to olccusion
            _target[k] = full_out.chunk(num_view, dim=0) # .view(num_view, batch_size, *full_out.shape[1:])

        if self.use_fake_K:
            K_mhmr = get_camera_parameters(self.mhmr_img_res, device=K.device) # if use pseudo K
            K_mhmr = K_mhmr.expand(K.shape[0], -1, -1)

        _target['smpl_mask'] = smpl_mask.chunk(num_view, dim=0)
        _target['K_mhmr'] = K_mhmr.chunk(num_view, dim=0)
        _target['img_mhmr'] = imgs_mhmr.chunk(num_view, dim=0)

        if "msk" in views[0]:
            msks = torch.stack([view["msk"] for view in views], dim=0)
            msks = msks.view(-1, *msks.shape[2:])
            msks_mhmr = pad_image(msks, self.mhmr_img_res, pad_value=0.0)  # bs,288,512->bs,896,896
            msks_mhmr = (msks_mhmr > 0.1).float()
            _target['msk_mhmr'] = msks_mhmr.chunk(num_view, dim=0)

        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))

        torch.cuda.empty_cache()


def get_patch_uv(imgshape, patch_size, pk_loc):
    n_patch = imgshape // patch_size  # H, W
    pk_idx = (pk_loc // patch_size).int()
    return n_patch, pk_idx

def get_score(n_patch, pk_idx, smpl_mask):
    # Scores & updating valid_humans according to occlusion - wap X and Y for scores only
    idx_h = torch.where(smpl_mask)
    nhv = int(smpl_mask.sum())
    bs = smpl_mask.shape[0]
    device = smpl_mask.device

    if isinstance(n_patch, (int, float)):
        patch_h, patch_w = int(n_patch), int(n_patch)
    else:
        patch_h, patch_w = n_patch[0], n_patch[1]

    scores = torch.zeros((bs, patch_h, patch_w)).to(device)
    visible_humans = torch.ones(nhv).to(device) # by default no occlusion so all visible

    for k in range(nhv):
        i = int(idx_h[0][k]) # index of the image
        j = int(idx_h[1][k]) # index of the human in this image
        _x = pk_idx[k,1] # patch center H
        _y = pk_idx[k,0] # patch center W
        # filter out heads out of cropping bounds
        if _x >= 0 and _x < patch_h and _y >= 0 and _y < patch_w:
            if scores[i,_x,_y] == 1:
                smpl_mask[i,j] = 0
                visible_humans[k] = 0
            else:
                scores[i,_x,_y] = 1
        else:
            smpl_mask[i,j] = 0
            visible_humans[k] = 0
    
    return smpl_mask, visible_humans, scores


import torch.nn as nn
from croco.models.blocks import Mlp_flex

class SMPLDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        target_dim=1,
        mlp_ratio=1,
        num_layers=2,
    ):
        super().__init__()
        self.mlp = Mlp_flex(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=target_dim,
            num_layers=num_layers,
            drop=0,
        )

    def forward(
        self,
        feat,
    ):
        """
        feat: BxC
        """

        pred = self.mlp(feat)
        return pred


def regression_mlp(layers_sizes):
    """
    Return a fully connected network.
    """
    assert len(layers_sizes) >= 2
    in_features = layers_sizes[0]
    layers = []
    for i in range(1, len(layers_sizes)-1):
        out_features = layers_sizes[i]
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, layers_sizes[-1]))
    return torch.nn.Sequential(*layers)

def apply_threshold(det_thresh, _scores):
    """ Apply thresholding to detection scores; if stack_K is used and det_thresh is a list, apply to each channel separately """
    if isinstance(det_thresh, list):
        det_thresh = det_thresh[0]
    idx = torch.where(_scores >= det_thresh)
    return idx

def nms(heat, kernel=3):
    """ easy non maximal supression (as in CenterNet) """

    if kernel not in [2, 4]:
        pad = (kernel - 1) // 2
    else:
        if kernel == 2:
            pad = 1
        else:
            pad = 2

    hmax = nn.functional.max_pool2d( heat, (kernel, kernel), stride=1, padding=pad)

    if hmax.shape[2] > heat.shape[2]:
        hmax = hmax[:, :, :heat.shape[2], :heat.shape[3]]

    keep = (hmax == heat).float()

    return heat * keep
