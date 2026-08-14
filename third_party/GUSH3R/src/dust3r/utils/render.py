import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from gsplat import rasterization
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.image import unpad_image
import numpy as np
try:
    import pyrender
except:
    import pyrender

import trimesh
from PIL import Image

def render(
    intrinsics: torch.Tensor,
    pts3d: torch.Tensor,
    rgbs: torch.Tensor | None = None,
    scale: float = 0.002,
    opacity: float = 0.95,
):

    device = pts3d.device
    batch_size = len(intrinsics)
    img_size = pts3d.shape[1:3]
    pts3d = pts3d.reshape(batch_size, -1, 3)
    num_pts = pts3d.shape[1]
    quats = torch.randn((num_pts, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = scale * torch.ones((num_pts, 3), device=device)
    opacities = opacity * torch.ones((num_pts), device=device)
    if rgbs is not None:
        assert rgbs.shape[1] == 3
        rgbs = rgbs.reshape(batch_size, 3, -1).transpose(1, 2)
    else:
        rgbs = torch.ones_like(pts3d[:, :, :3])

    rendered_rgbs = []
    rendered_depths = []
    accs = []
    for i in range(batch_size):
        rgbd, acc, _ = rasterization(
            pts3d[i],
            quats,
            scales,
            opacities,
            rgbs[i],
            torch.eye(4, device=device)[None],
            intrinsics[[i]],
            width=img_size[1],
            height=img_size[0],
            packed=False,
            render_mode="RGB+D",
        )

        rendered_depths.append(rgbd[..., 3])

    rendered_depths = torch.cat(rendered_depths, dim=0)

    return rendered_rgbs, rendered_depths, accs


def get_render_results(gts, preds, self_view=False):
    device = preds[0]["pts3d_in_self_view"].device
    with torch.no_grad():
        depths = []
        gt_depths = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if self_view:
                camera = inv(gt["camera_pose"]).to(device)
                intrinsics = gt["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_self_view"]
            else:
                camera = inv(gts[0]["camera_pose"]).to(device)
                intrinsics = gts[0]["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_other_view"]
            gt_img = gt["img"].to(device)
            gt_pts3d = gt["pts3d"].to(device)

            _, depth, _ = render(intrinsics, pred, gt_img)
            _, gt_depth, _ = render(intrinsics, geotrf(camera, gt_pts3d), gt_img)
            depths.append(depth)
            gt_depths.append(gt_depth)
    return depths, gt_depths


def vis_heatmap(img, scores):
    hm = scores.clone()
    hm = torch.clamp(hm + 0.1, 0, 1)  # for visu purpose only
    hm = hm.unsqueeze(0).unsqueeze(0)
    hm = torch.nn.functional.interpolate(
        hm, 
        size=(img.shape[0], img.shape[1]),
        mode='nearest'
    ).squeeze(0).squeeze(0)
    
    hm = hm.unsqueeze(-1)
    
    return img * hm

def get_render_smpl(gts, preds, smpl_model, loss_details, has_msk=False):
    with torch.no_grad():
        smpl_faces = {
            'neutral': {
                10: smpl_model.smplx_neutral_10.faces,
                11: smpl_model.smplx_neutral_11.faces,
            }
        }
    
        gt_hms_list, pr_hms_list, gt_smpls_list, pr_smpls_list= [], [], [], []
        gt_msks_list, pr_msks_list = [], []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            gt_img = gt["img"]
            smpl_mask = gt["smpl_mask"]
            K = gt["camera_intrinsics"]

            idx_h = torch.where(smpl_mask)
            if has_msk:
                gt_msk, pr_msk = gt["msk_mhmr"], pred["msk"][...,0]
                gt_msk = unpad_image(gt_msk[:,None], gt_img.shape[2:])[:, 0]
                pr_msk = unpad_image(pr_msk[:,None], gt_img.shape[2:])[:, 0]
            gt_scores, pr_scores = gt["smpl_scores"], pred["smpl_scores"][...,0]
            gt_scores = unpad_image(gt_scores[:,None], gt_img.shape[2:])[:, 0] # if use K of CUT3R, unpad the scores
            pr_scores = unpad_image(pr_scores[:,None], gt_img.shape[2:])[:, 0]
            
            has_smpl_mesh = (
                int(smpl_mask.sum()) > 0
                and "smpl_shape" in gt
                and "smpl_v3d" in gt
                and "smpl_shape" in pred
            )
            if has_smpl_mesh:
                gt_shape = gt["smpl_shape"].shape[-1]
                pr_shape = pred["smpl_shape"].shape[-1]
                gt_v3d = gt["smpl_v3d"][smpl_mask]
                pr_v3d = loss_details[f"pred_smpl_v3d_{i+1}"][smpl_mask]
            
            gt_hms, pr_hms, gt_msks, pr_msks, gt_smpls, pr_smpls= [], [], [], [], [], []
            for k in range(len(gt_img)):
                img_array = (0.5 * (gt_img[k] + 1.0)).permute(1, 2, 0)

                if has_msk:
                    gt_msk_array = vis_heatmap(img_array, gt_msk[k])
                    pr_msk_array = vis_heatmap(img_array, pr_msk[k])

                gt_hm_array = vis_heatmap(img_array, gt_scores[k])
                pr_hm_array = vis_heatmap(img_array, pr_scores[k])

                img_array_np = (img_array * 255).cpu().numpy().astype(np.uint8)
                focal = K[k,[0,1],[0,1]].cpu().numpy()
                princpt = K[k,[0,1],[-1,-1]].cpu().numpy()
                gt_verts, gt_faces, pr_verts, pr_faces = [], [], [], []
                if has_smpl_mesh:
                    for j in range(len(idx_h[0])):
                        if idx_h[0][j] == k:
                            gt_verts.append(gt_v3d[j].detach().cpu().numpy().reshape(-1,3))
                            gt_faces.append(smpl_faces["neutral"][gt_shape])
                            pr_verts.append(pr_v3d[j].detach().cpu().numpy().reshape(-1,3))
                            pr_faces.append(smpl_faces["neutral"][pr_shape])
                    gt_rend_array = torch.as_tensor(
                        render_meshes(img_array_np.copy(), 
                        gt_verts, gt_faces,
                        {'focal': focal, 'princpt': princpt}),
                        ) / 255.0
                    pr_rend_array = torch.as_tensor(
                        render_meshes(img_array_np.copy(), 
                        pr_verts, pr_faces,
                        {'focal': focal, 'princpt': princpt}),
                        ) / 255.0
                else:
                    gt_rend_array = img_array.detach().cpu()
                    pr_rend_array = img_array.detach().cpu()

                gt_hms.append(gt_hm_array)
                pr_hms.append(pr_hm_array)
                gt_smpls.append(gt_rend_array)
                pr_smpls.append(pr_rend_array)
                if has_msk:
                    gt_msks.append(gt_msk_array)
                    pr_msks.append(pr_msk_array)
       
            gt_hms_list.append(torch.stack(gt_hms, 0))
            pr_hms_list.append(torch.stack(pr_hms, 0))
            gt_smpls_list.append(torch.stack(gt_smpls, 0))
            pr_smpls_list.append(torch.stack(pr_smpls, 0))
            if has_msk:
                gt_msks_list.append(torch.stack(gt_msks, 0))
                pr_msks_list.append(torch.stack(pr_msks, 0)) 
    return (
        gt_msks_list, pr_msks_list, 
        gt_hms_list, pr_hms_list, 
        gt_smpls_list, pr_smpls_list
        )


OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]])

def render_meshes(img, l_mesh, l_face, cam_param, color=None, alpha=1.0, 
                  show_camera=False,
                  intensity=3.0,
                  metallicFactor=0., roughnessFactor=0.5, smooth=True,
                  ):
    """
    Rendering multiple mesh and project then in the initial image.
    Args:
        - img: np.array [w,h,3]
        - l_mesh: np.array list of [v,3]
        - l_face: np.array list of [f,3]
        - cam_param: info about the camera intrinsics (focal, princpt) and (R,t) is possible
    Return:
        - img: np.array [w,h,3]
    """
    # scene
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))

    # mesh
    for i, mesh in enumerate(l_mesh):
        if color is None:
            _color = (np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255)
        else:
            if isinstance(color,list):
                _color = color[i]
            elif isinstance(color,tuple):
                _color = color
            else:
                raise NotImplementedError
        mesh = trimesh.Trimesh(mesh, l_face[i])
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=metallicFactor,
            roughnessFactor=roughnessFactor,
            alphaMode='OPAQUE',
            baseColorFactor=(_color[0], _color[1], _color[2], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
        scene.add(mesh, f"mesh_{i}")

    # Adding coordinate system at (0,0,2) for the moment
    # Using lines defined in pyramid https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Pyramid.html
    if show_camera:
        import pyvista

        def get_faces(x):
            return x.faces.astype(np.uint32).reshape((x.n_faces, 4))[:, 1:]
        
        # Camera = Box + Cone (or Cylinder?)
        material_cam = pyrender.MetallicRoughnessMaterial(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor, alphaMode='OPAQUE', baseColorFactor=(0.5,0.5,0.5))
        height = 0.2
        radius = 0.1
        cone = pyvista.Cone(center=(0.0, 0.0, -height/2), direction=(0.0, 0.0, -1.0), height=height, radius=radius).extract_surface().triangulate()
        verts = cone.points
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, get_faces(cone)), material=material_cam, smooth=smooth)
        scene.add(mesh, f"cone")

        size = 0.1
        box = pyvista.Box(bounds=(-size, size, 
                                  -size, size, 
                                  verts[:,-1].min() - 3*size, verts[:,-1].min())).extract_surface().triangulate()
        verts = box.points
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, get_faces(box)), material=material_cam, smooth=smooth)
        scene.add(mesh, f"box")
        

        # Coordinate system
        # https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Arrow.html
        l_color = [(1,0,0,1.0), (0,1,0,1.0), (0,0,1,1.0)]
        l_direction = [(1,0,0), (0,1,0), (0,0,1)]
        scale = 0.2
        pose3d = [2*scale, 0.0, -scale]
        for i in range(len(l_color)):
            arrow = pyvista.Arrow(direction=l_direction[i], scale=scale)
            arrow = arrow.extract_surface().triangulate()
            verts = arrow.points + np.asarray([pose3d])
            faces = arrow.faces.astype(np.uint32).reshape((arrow.n_faces, 4))[:, 1:]
            mesh = trimesh.Trimesh(verts, faces)
            material = pyrender.MetallicRoughnessMaterial(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor, alphaMode='OPAQUE', baseColorFactor=l_color[i])
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
            scene.add(mesh, f"arrow_{i}")
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera_pose = np.eye(4)
    if 'R' in cam_param.keys():
        camera_pose[:3, :3] = cam_param['R']
    if 't' in cam_param.keys():
        camera_pose[:3, 3] = cam_param['t']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    
    # camera
    camera_pose = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ camera_pose
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(intensity=intensity)
    scene.add(light, pose=camera_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    fg = (depth > 0)[:,:,None].astype(np.float32)

    # Simple smoothing of the mask
    bg_blending_radius = 1
    bg_blending_kernel = 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (2 * bg_blending_radius + 1) ** 2
    bg_blending_bias =  -torch.ones(1)
    fg = fg.reshape((fg.shape[0],fg.shape[1]))
    fg = torch.from_numpy(fg).unsqueeze(0)
    fg = torch.clamp_min(torch.nn.functional.conv2d(fg, weight=bg_blending_kernel, bias=bg_blending_bias, padding=bg_blending_radius) * fg, 0.0)
    fg = fg.permute(1,2,0).numpy()

    # Alpha-blending
    img = (fg * (alpha * rgb + (1.0-alpha) * img) + (1-fg) * img).astype(np.uint8)

    renderer.delete()

    return img.astype(np.uint8)
