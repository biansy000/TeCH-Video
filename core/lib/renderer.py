import os
import math
import cv2
import trimesh
import numpy as np
import random
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import nvdiffrast.torch as dr
# import kaolin as kal
from encoding import get_encoder
# # from meshutils import remesh
from .texture import Texture3D

from .obj import Mesh, safe_normalize
from .marching_tets import DMTet
from .build_tet_grid import build_tet_grid, build_tet_grid_new
from .dmtet_network import DMTetMesh
from .dmtet_network_restpose import DMTetMeshRestPose
from .dmtet_network_wmask import DMTetMeshWMask
from .uv_utils import texture_padding
from PIL import Image
def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (
        x.shape[1] < size[0] and
        x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(
                nn.Linear(
                    self.dim_in if l == 0 else self.dim_hidden,
                    self.dim_out if l == num_layers - 1 else self.dim_hidden,
                    bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class Renderer(nn.Module):

    def __init__(
        self,
        opt,
        smpl_v_rest=None,
        smpl_faces=None,
        # smpl_v=None,
        rest_T=None,
        smpl_v_T=None,
        joints_rest=None,
        num_layers_bg=2,
        hidden_dim_bg=16,
    ):

        super().__init__()

        self.opt = opt
        self.min_near = opt.min_near
        # self.v_offsets = 0
        # self.vn_offsets = 0

        if not self.opt.use_gl:
            self.glctx = dr.RasterizeCudaContext()  # support at most 2048 resolution.
        else:
            print('building gl context')
            # # try:
            self.glctx = dr.RasterizeGLContext()  # will crash if using GUI...
            # except:
            # print('Failed to initialize GLContext, use CudaContext instead...')
            # self.glctx = dr.RasterizeCudaContext()  # support at most 2048 resolution.
        # load the template mesh, will calculate normal and texture if not provided.
        self.texture3d = Texture3D(opt)
        
        # TODO: textrue 2D

        self.use_pose_control = opt.use_pose_control

        if opt.mesh_resolution > 0: # geo: False
            tet_file = Path(f'data/tets/{opt.mesh_resolution}_tets.npz').expanduser()
            tets = np.load(tet_file)
            print("Loaded base mesh file", tet_file)
            self.mesh = Mesh(v=torch.tensor(tets['vertices']) * 2.0, f=torch.tensor(tets['indices'], dtype=torch.int32))
            self.marching_tets = DMTet()
            self.dmtet_network = None

        elif opt.progressive_geo:
            self.mesh = Mesh.load_obj(
                self.opt.last_model, ref_path=self.opt.last_ref_model, init_empty_tex=self.opt.init_empty_tex, albedo_res=self.opt.albedo_res, 
                keypoints_path=self.opt.keypoints_path, init_uv=False)
            if self.mesh.keypoints is not None:
                self.keypoints = self.mesh.keypoints
            else:
                self.keypoints = None
            self.marching_tets = None
            tet_v, tet_ind = build_tet_grid(self.mesh, opt)
            self.dmtet_network = DMTetMeshWMask(vertices=torch.tensor(tet_v, dtype=torch.float), indices=torch.tensor(tet_ind, dtype=torch.long), grid_scale=self.opt.tet_grid_scale, use_explicit=opt.use_explicit_tet, geo_network=opt.dmtet_network, 
                                           hash_max_res=opt.geo_hash_max_res, hash_num_levels=opt.geo_hash_num_levels, num_subdiv=opt.tet_num_subdiv)
            if self.opt.init_mesh and not self.opt.test:
                self.dmtet_network.init_mesh(self.mesh.v, self.mesh.f, self.opt.init_mesh_padding)
        elif opt.use_dmtet_posed: # geo: True
            self.mesh = Mesh.load_obj(self.opt.last_model.replace('.obj', '_rest.obj'), 
                                      ref_path=self.opt.last_ref_model, init_empty_tex=self.opt.init_empty_tex,
                                      albedo_res=self.opt.albedo_res, keypoints_path=self.opt.keypoints_path, init_uv=False)
            
            resize_matrix_inv = self.mesh.resize_matrix_inv
            resize_matrix = resize_matrix_inv.inverse()

            # print('resize_matrix', resize_matrix, resize_matrix_inv)
            # change to the normalized coordinate system (
            smpl_v_rest_homo = torch.cat(
                [smpl_v_rest, torch.ones_like(smpl_v_rest)[:, :, :1]], dim=-1)
            self.register_buffer('smpl_v_rest', torch.einsum('ij,bkj->bki', resize_matrix, smpl_v_rest_homo)[:, :, :3])
            self.register_buffer('smpl_faces', smpl_faces)
            self.register_buffer('rest_T', resize_matrix @ rest_T @ resize_matrix_inv)
            self.register_buffer('smpl_v_T', resize_matrix @ smpl_v_T @ resize_matrix_inv)
            joints_rest_homo = torch.cat(
                [joints_rest, torch.ones_like(joints_rest)[:, :, :1]], dim=-1
            )
            self.register_buffer('joints_rest', torch.einsum('ij,bkj->bki', resize_matrix, joints_rest_homo)[:, :, :3])

            # assert False

            if self.mesh.keypoints is not None:
                self.keypoints = self.mesh.keypoints
            else:
                self.keypoints = None
            self.marching_tets = None
            tet_v, tet_ind = build_tet_grid_new(self.mesh, opt, smpl_rest_v=smpl_v_rest[0], smpl_faces=smpl_faces[0])
            self.dmtet_network = DMTetMeshRestPose(vertices=torch.tensor(tet_v, dtype=torch.float), indices=torch.tensor(tet_ind, dtype=torch.long), grid_scale=self.opt.tet_grid_scale, use_explicit=opt.use_explicit_tet, geo_network=opt.dmtet_network, 
                                           hash_max_res=opt.geo_hash_max_res, hash_num_levels=opt.geo_hash_num_levels, num_subdiv=opt.tet_num_subdiv)
            if self.opt.init_mesh and not self.opt.test:
                self.dmtet_network.init_mesh(self.mesh.v, self.mesh.f, 
                                             smpl_v_rest=self.smpl_v_rest, smpl_faces=self.smpl_faces, rest_T=self.rest_T, smpl_v_T=self.smpl_v_T, 
                                             init_padding=self.opt.init_mesh_padding)
        
        elif opt.use_dmtet_new: # geo: True
            self.mesh = Mesh.load_obj(
                self.opt.last_model, ref_path=self.opt.last_ref_model, init_empty_tex=self.opt.init_empty_tex, albedo_res=self.opt.albedo_res, 
                keypoints_path=self.opt.keypoints_path, init_uv=False)
            if self.mesh.keypoints is not None:
                self.keypoints = self.mesh.keypoints
            else:
                self.keypoints = None
            self.marching_tets = None
            tet_v, tet_ind = build_tet_grid(self.mesh, opt)
            self.dmtet_network = DMTetMesh(vertices=torch.tensor(tet_v, dtype=torch.float), indices=torch.tensor(tet_ind, dtype=torch.long), grid_scale=self.opt.tet_grid_scale, use_explicit=opt.use_explicit_tet, geo_network=opt.dmtet_network, 
                                           hash_max_res=opt.geo_hash_max_res, hash_num_levels=opt.geo_hash_num_levels, num_subdiv=opt.tet_num_subdiv)
            if self.opt.init_mesh and not self.opt.test:
                self.dmtet_network.init_mesh(self.mesh.v, self.mesh.f, self.opt.init_mesh_padding)

        else:
            self.mesh = Mesh.load_obj(self.opt.last_model, ref_path=self.opt.last_ref_model, init_empty_tex=self.opt.init_empty_tex, albedo_res=self.opt.albedo_res, use_vertex_tex=self.opt.use_vertex_tex, keypoints_path=self.opt.keypoints_path, init_uv=self.opt.save_mesh)
            if self.mesh.keypoints is not None:
                self.keypoints = self.mesh.keypoints
            else:
                self.keypoints = None
            self.marching_tets = None
            self.dmtet_network = None
        
        self.use_dmtet_posed = opt.use_dmtet_posed

        self.mesh.v = self.mesh.v * self.opt.mesh_scale
        if opt.init_texture_3d:
            self.init_texture_3d()

        if opt.use_vertex_tex:
            self.vertex_albedo = nn.Parameter(self.mesh.v_color)
        # extract trainable parameters
        if self.dmtet_network is None and not opt.lock_geo:
            self.sdf = nn.Parameter(torch.zeros_like(self.mesh.v[..., 0]))
            self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
            self.vn_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
        
        if self.opt.can_pose_folder:
            import glob
            if '.obj' in self.opt.can_pose_folder:
                can_pose_objs = [self.opt.can_pose_folder]
            else:    
                can_pose_objs = glob.glob(self.opt.can_pose_folder + '/*.obj')
            self.can_pose_vertices = []
            self.can_pose_faces = []
            self.can_pose_resize_inv = []
            for pose_obj in can_pose_objs:
                tri_mesh = trimesh.load(pose_obj)
                mesh = Mesh(torch.tensor(tri_mesh.vertices, dtype=torch.float32).cuda(), torch.tensor(tri_mesh.faces, dtype=torch.int32).cuda())
                mesh.auto_size()
                self.can_pose_vertices.append(mesh.v)
                self.can_pose_faces.append(mesh.f)
                

        # background network
        self.encoder_bg, self.in_dim_bg = get_encoder('frequency_torch', input_dim=3, multires=4)
        self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
        if self.opt.different_bg:
            self.normal_bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            self.textureless_bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
        # NOTE: similiar bg_net in texture3d

        # TEST: a hashgrid color encoder for color
        # self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048, interpolation='smoothstep')
        # self.color_net = MLP(self.in_dim, 3, 32, 2, bias=True)

    def init_texture_3d(self):
        self.texture3d = self.texture3d.cuda()
        optimizer = torch.optim.Adam(self.texture3d.parameters(), lr=0.01, betas=(0.9, 0.99),
                                            eps=1e-15)
        os.makedirs(self.opt.workspace, exist_ok=True)
        ckpt_path = os.path.join(self.opt.workspace, 'init_tex.pth')
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            self.texture3d.load_state_dict(state_dict)
        elif self.mesh.v_color is not None:
            batch_size = 300000
            num_epochs = 200
            v_norm = self.mesh.vn
            v_color = self.mesh.v_color
            v_pos = self.mesh.v
            num_pts = v_pos.shape[0]
            print('start init texture 3d')
            for i in range(num_epochs):
                optimizer.zero_grad()
                indice = random.sample(range(num_pts), min(batch_size, num_pts))
                batch_pos = v_pos[indice].cuda()
                batch_norm = v_norm[indice].cuda()
                batch_color = v_color[indice].cuda()
                _, pred_color, pred_norm = self.texture3d(batch_pos, None, shading='output')
                loss_norm = nn.functional.mse_loss(pred_norm, batch_norm)
                loss_rgb = nn.functional.mse_loss(pred_color, batch_color)
                loss = loss_rgb + loss_norm
                loss.backward()
                optimizer.step()
                print('Iter {}: loss_norm: {}, loss_rgb: {}'.format(i, loss_norm.data, loss_rgb.data))
            torch.save(self.texture3d.state_dict(), ckpt_path)


    def get_initial_guess(self, init_normal=True):
        if self.opt.last_model.endswith('.obj') or self.opt.last_model.endswith('.ply'):
            return
        pth = torch.load(self.opt.last_model, map_location='cpu')
        self.texture3d.load_state_dict(pth['model'])
        print('Loaded last model from', self.opt.last_model)

        sigma, albedo, enc = self.texture3d.common_forward(self.mesh.v)
        self.sdf.data.copy_(sigma - min(pth['mean_density'], self.opt.density_thresh))  # like export_mesh

        if init_normal:
            normal = self.texture3d.normal_net(enc)
            normal = safe_normalize(normal)
            normal = torch.nan_to_num(normal)
            self.vn_offsets.data.copy_(normal)

    # optimizer utils
    def get_params(self, lr):
        # yapf: disable

        params = [
            # {'params': self.raw_albedo, 'lr': lr * 10},
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            # {'params': self.color_net.parameters(), 'lr': lr},
            {'params': self.bg_net.parameters(), 'lr': lr},
        ]
        if self.opt.different_bg:
            params += [
                {'params': self.textureless_bg_net.parameters(), 'lr': lr},
                {'params': self.normal_bg_net.parameters(), 'lr': lr},
            ]

        if self.texture3d is not None:
            params += [
                {'params': self.texture3d.encoder.parameters(),    'lr': lr},
                {'params': self.texture3d.sigma_net.parameters(),  'lr': lr},
                {'params': self.texture3d.normal_net.parameters(), 'lr': lr},
            ]

        if not self.opt.lock_geo:
            if self.dmtet_network is not None:
                params.extend([
                    {'params': self.dmtet_network.parameters(), 'lr': lr*self.opt.dmtet_lr}
                ])
            else:
                params.extend([
                    {'params': self.v_offsets, 'lr': 0.0001},
                    {'params': self.vn_offsets, 'lr': 0.0001},
                ])
        # yapf: enable
        if self.opt.use_vertex_tex:
            vertex_tex_lr = lr * 1
            params.extend([
                {'params': self.vertex_albedo, 'lr': vertex_tex_lr}
            ])
            print('vertex_tex_lr:', vertex_tex_lr)
        return params
    

    @torch.no_grad()
    def export_mesh(self, save_path, smpl_v_T=None, name='mesh', export_uv=False):
        self.resize_matrix_inv = self.mesh.resize_matrix_inv
        if self.dmtet_network is not None:
            num_subdiv = self.get_num_subdiv()
            with torch.no_grad():
                # verts, faces, loss = self.dmtet_network.get_mesh(return_loss=False, num_subdiv=num_subdiv)
                if self.use_dmtet_posed:
                #     verts, faces, loss = self.dmtet_network.get_mesh(return_loss=False, num_subdiv=num_subdiv)
                # else:
                    verts, verts_rest, faces, loss = self.dmtet_network.get_mesh(
                        smpl_v_rest=self.smpl_v_rest, smpl_faces=self.smpl_faces, rest_T=self.rest_T, smpl_v_T=self.smpl_v_T,
                        return_loss=False, num_subdiv=num_subdiv)
                elif self.opt.progressive_geo:
                    verts, faces, loss = self.dmtet_network.get_mesh(return_loss=False, num_subdiv=num_subdiv, global_step=self.global_step)
                else:
                    verts, faces, loss = self.dmtet_network.get_mesh(return_loss=False, num_subdiv=num_subdiv)
            
            self.mesh = Mesh(v=verts, f=faces.int(), device='cuda', split=True)
            self.mesh.albedo = torch.ones((2048, 2048, 3), dtype=torch.float).cuda()
            if export_uv:
                self.mesh.auto_uv()
                self.mesh.auto_normal()
        elif hasattr(self, 'v_offsets') and hasattr(self, 'vn_offsets'):
            self.mesh.v = (self.mesh.v + self.v_offsets).detach()
            self.mesh.vn = (self.mesh.vn + self.vn_offsets).detach()  # TODO: may not be unit ?
        else:
            self.mesh.v = self.mesh.v
            self.mesh.vn = self.mesh.vn
        if export_uv:
            if self.opt.use_vertex_tex:
                self.mesh.v_color = self.vertex_albedo.detach().clamp(0, 1)
            elif self.opt.use_texture_2d:
                self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
            else:
                self.mesh.albedo = self.get_albedo_from_texture3d()
        verts = torch.cat([self.mesh.v, torch.ones_like(self.mesh.v[:, :1])], dim=1) @ self.resize_matrix_inv.T
        self.mesh.v = verts
        self.mesh.write(os.path.join(save_path, '{}.obj'.format(name)))
        if self.opt.da_pose_mesh:
            import trimesh
            verts = self.mesh.v.new_tensor(trimesh.load(self.opt.da_pose_mesh).vertices)
            assert verts.shape[0] == self.mesh.v.shape[0], f"pose mesh verts: {self.mesh.v.shape[0]}, da pose mesh verts: {verts.shape[0]}"
            self.mesh.v = verts
            self.mesh.write(os.path.join(save_path, '{}_da.obj'.format(name)))

    @torch.no_grad()
    def get_front_view_triangles(self, h, w):
        if self.opt.mesh_dataset == 'phorhum':
            TO_WORLD = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -0.98226033, -0.18752234, 0.5],
                    [0.0, 0.18752234, -0.98226033, 2.2],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            far = 1000
            near = self.opt.min_near
            projection = torch.tensor([
                [2 , 0, 0, 0], 
                [0, 2 , 0, 0],
                [0, 0, (far + near)/(far - near), -(2 * far * near)/(far - near)], 
                [0, 0, 1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0) # yapf: disabl
            mvp = projection @ torch.tensor(np.linalg.inv(TO_WORLD)).cuda() @ self.resize_matrix_inv
        else:
            TO_WORLD = np.eye(
                4,
                dtype=np.float32,
            )
            TO_WORLD[2,2] = -1
            TO_WORLD[1,1] = -1
            mvp = torch.tensor(np.linalg.inv(TO_WORLD)).cuda() @ self.resize_matrix_inv
        v = self.mesh.v  # [N, 3]
        v_clip = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0),
                              torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))
        frontview_triangles = torch.unique(rast[..., 3].view(-1)).to(torch.long)
        return frontview_triangles
        
    
    @torch.no_grad()
    def get_albedo_from_texture3d(self):
        h, w = self.mesh.albedo.shape[:2]
        uv = self.mesh.vt *2.0 - 1.0
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
        print(uv.shape, self.mesh.ft.shape, h, w)
        rast, rastdb = dr.rasterize(self.glctx, uv.unsqueeze(0), self.mesh.ft, (h, w)) # [1, h, w, 4]
        
        if not self.opt.use_can_pose_space:
            color_space_v, color_space_f = self.mesh.v, self.mesh.f
        else:
            color_space_v, color_space_f = self.can_pose_vertices[0], self.can_pose_faces[0]

        xyzs, _ = dr.interpolate(color_space_v.unsqueeze(0), rast, color_space_f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(self.mesh.v[:, :1]).unsqueeze(0), rast, self.mesh.f) # [1, h, w, 1]
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        #Image.fromarray((mask.reshape(h, w).cpu().numpy()*255).astype(np.uint8)).save('uv_map_mask.png')
        feats = torch.zeros(h * w, 3, device='cuda', dtype=torch.float32)
        batch_size = 300000
        xyzs = xyzs[mask]
        num_pts = xyzs.shape[0]
        res = []
        for i in range(0, num_pts, batch_size):
            i_end = min(i + batch_size, num_pts)
            batch_pts = xyzs[i:i_end]
            _, pred_color, pred_norm = self.texture3d(batch_pts, None, shading='output')
            res.append(pred_color)
        mask_feats = torch.cat(res, dim=0)
        print(feats.shape, mask.shape, mask_feats.shape)
        feats[mask] = mask_feats
        feats = feats.reshape(h, w, 3)
        if self.opt.color_correction:
            from skimage import exposure
            tex_triangles = rast[0, :, :, 3].view(-1).to(torch.long) - 1
            rast_mask = rast[0, :, :, 3] > 0
            front_view_triangles = self.get_front_view_triangles(h,w) - 1
            print(tex_triangles.max(), tex_triangles.min())
            front_tri_mask = torch.zeros_like(self.mesh.ft[..., 0]).to(torch.bool).cuda()
            print(front_tri_mask.shape)
            front_tri_mask[front_view_triangles] = True
            tex_front_mask = front_tri_mask[tex_triangles].reshape(h, w)
            #Image.fromarray((((~tex_front_mask)&(rast_mask)).detach().cpu().numpy()*255).astype(np.uint8)).save('tex_front_mask.png')
            generated = (feats[(~tex_front_mask)&(rast_mask)].detach().cpu().numpy() * 255).astype(np.uint8)
            reference = (feats[tex_front_mask & rast_mask].detach().cpu().numpy() * 255).astype(np.uint8)
            generated = exposure.match_histograms(generated, reference, channel_axis=-1)
            feats[(~tex_front_mask)&(rast_mask)] = feats.new_tensor(generated/255)
        feats = self.mesh.albedo.new_tensor(texture_padding((feats.reshape(h, w, 3).cpu().numpy()*255).astype(np.uint8), (mask.reshape(h, w).cpu().numpy()*255).astype(np.uint8))) / 255

        return feats.reshape(self.mesh.albedo.shape)

    @torch.no_grad()
    def remesh(self):

        device = self.mesh.v.device
        if hasattr(self, 'v_offsets'):
            v = (self.mesh.v + self.v_offsets).detach().cpu().numpy()
        else:
            v = self.mesh.v.detach().cpu().numpy()
        f = self.mesh.f.detach().cpu().numpy()

        v, f = remesh(v, f, remesh_size=self.opt.remesh_size)

        self.mesh.v = torch.from_numpy(v).float().contiguous().to(device)  # [N, 3]
        self.mesh.f = torch.from_numpy(f).int().contiguous().to(device)
        self.mesh.auto_normal()

        # TODO: how to keep UV while doing remesh ???
        raise NotImplementedError

        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
        self.vn_offsets = nn.Parameter(torch.zeros_like(self.mesh.vn))

        print(f'[INFO] remesh: {self.mesh.v.shape}, {self.mesh.f.shape}')

    def get_mesh(self, return_loss=True, detach_geo=False, global_step=1e7):
        if self.marching_tets is None and self.dmtet_network is None:
            #return Mesh(v=self.mesh.v+self.v_offsets, base=self.mesh, device='cuda'), None
            return Mesh(v=self.mesh.v, base=self.mesh, device='cuda'), None
        loss = None
        if self.opt.use_dmtet_new or self.use_dmtet_posed:                
            num_subdiv = self.get_num_subdiv(global_step=global_step)

            if self.use_dmtet_posed:
            #     verts, faces, loss = self.dmtet_network.get_mesh(return_loss=return_loss, num_subdiv=num_subdiv)
            # else:
                verts, verts_rest, faces, loss = self.dmtet_network.get_mesh(
                    smpl_v_rest=self.smpl_v_rest, smpl_faces=self.smpl_faces, rest_T=self.rest_T, smpl_v_T=self.smpl_v_T,
                    return_loss=return_loss, num_subdiv=num_subdiv)
            elif self.opt.progressive_geo:
                verts, faces, loss = self.dmtet_network.get_mesh(return_loss=return_loss, num_subdiv=num_subdiv, global_step=global_step)
            else:
                verts, faces, loss = self.dmtet_network.get_mesh(return_loss=return_loss, num_subdiv=num_subdiv)

            if detach_geo:
                verts = verts.detach()
                faces = faces.detach()
                loss = None
            mesh = Mesh(v=verts, f=faces.int(), device='cuda')
        else:
            v_deformed = self.mesh.v
            if hasattr(self, 'v_offsets'):
                v_deformed = v_deformed + 2 / (self.opt.mesh_resolution * 2) * torch.tanh(self.v_offsets)
            verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf, self.mesh.f)
            mesh = Mesh(v=verts, f=faces.int(), vt=uvs, ft=uv_idx.int())
        #('verts.grad: {} mesh.v.grad: {}'.format(verts.requires_grad, mesh.v.requires_grad))
        mesh.auto_normal()
    
        return mesh, loss
    
    def get_color_from_vertex_texture(self, rast, rast_db, f, light_d, ambient_ratio, shading) -> Tensor:
        albedo, _ = dr.interpolate(
            self.vertex_albedo.unsqueeze(0).contiguous(), rast, f, rast_db=rast_db)
        albedo = albedo.clamp(0., 1.)
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0.).to(albedo.device))  # remove background

        if shading == 'albedo':
            normal = None
            color = albedo
        else:
            # NOTE: normal is hard... since we allow v to change, we have to recalculate normals all the time! and must care about flipped faces...
            vn = self.mesh.vn
            if hasattr(self, 'vn_offsets'):
                vn = vn + self.vn_offsets
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.f)
            normal = safe_normalize(normal)

            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d).float().clamp(min=0)

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
        return color

    def get_color_from_mesh(self, mesh, rast, light_d, ambient_ratio, shading, poses=None):
        vn = mesh.vn
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, mesh.f)
        normal = safe_normalize(normal)
        #print('vn.grad {} normal.grad {}'.format(vn.requires_grad, normal is not None))
        lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d).float().clamp(min=0)

        if shading == 'textureless':
            color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif shading == 'normal':
            if self.opt.render_relative_normal and poses is not None:
                normal_shape_old = normal.shape
                B = poses.shape[0]
                normal = torch.matmul(F.pad(normal, pad=(0, 1), mode='constant', value=1.0).reshape(B, -1, 4),
                              torch.transpose(torch.inverse(poses.cpu()).to(normal.device), 1, 2).reshape(B, 4, 4)).float()
                normal = normal[..., :3].reshape(normal_shape_old)
                normal = normal * normal.new_tensor([1, 1, -1])
            color = (normal + 1) / 2
        return color

    def get_color_from_2d_texture(self, rast, rast_db, mesh, rays_o, light_d, ambient_ratio, shading) -> Tensor:
        texc, texc_db = dr.interpolate(
            self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(
            self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [1, H, W, 3]
        # texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft)
        # albedo = dr.texture(self.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
        albedo = torch.sigmoid(albedo)
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background

        if shading == 'albedo':
            normal = None
            color = albedo

        else:

            # NOTE: normal is hard... since we allow v to change, we have to recalculate normals all the time! and must care about flipped faces...
            vn = mesh.vn
            if hasattr(self, 'vn_offsets'):
                vn = vn + self.vn_offsets
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, mesh.f)
            normal = safe_normalize(normal)

            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d).float().clamp(min=0)

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
        return color

    def get_color_from_3d_texture(self, rast, rast_db, v, f, vn, light_d, ambient_ratio, shading) -> Tensor:
        xyzs, _ = dr.interpolate(v, rast, f, rast_db)
        sigma, albedo, _ = self.texture3d(xyzs.view(-1, 3), None, light_d, ambient_ratio, 'albedo')
        if vn is not None:
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, f)
            normal = safe_normalize(normal)
        if shading == 'albedo':
            normal = None
            color = albedo
        else:
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d).float().clamp(min=0)
            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.reshape(albedo.shape[:-1]).unsqueeze(-1)
        return color.view(*rast.shape[:-1], 3)
    
    def get_can_pos_map(self, rast, rast_db, f) -> Tensor:
        #print(self.can_pose_vertices[0].shape)
        xyzs, _ = dr.interpolate(self.can_pose_vertices[0], rast, f, rast_db)
        #print(xyzs.shape)
        return xyzs
        
    def get_openpose_map(self, keypoints, depth, rgb=None):
        keypoints = keypoints[0, 0]
        keypoints = keypoints[:, :3] / keypoints[:, 3:]
        from .pose_utils import draw_openpose_map
        # print('depth.shape', depth.shape)
        # print('keypoints', keypoints)
        # print('depth.max()', depth.max())
        H, W = depth.shape[:2]
        keypoints_2d = (keypoints[:, :2] + 1.) / 2
        keypoints_depth = keypoints[:, 2]
        keypoints_2d_int = (keypoints_2d.clamp(0, 1.) * keypoints_2d.new_tensor([W, H])).to(torch.int)
        keypoints_2d_int[:, 0] = keypoints_2d_int[:, 0].clamp(0, W-1)
        keypoints_2d_int[:, 1] = keypoints_2d_int[:, 1].clamp(0, H-1)
        keypoints_depth_proj = torch.zeros_like(keypoints_depth)
        for i in range(len(keypoints_2d_int)):
            keypoints_depth_proj[i] = depth[keypoints_2d_int[i, 1], keypoints_2d_int[i, 0], 0]
        depth_diff_thres = (keypoints_depth[56:56+68].max(dim=0)[0] - keypoints_depth[56:56+68].min(dim=0)[0])/5
        #print(depth_diff_thres)
        keypoints_mask = (keypoints_2d[:, 0] < 1) & (keypoints_2d[:, 0] >= 0) & (keypoints_2d[:, 1] < 1) & (keypoints_2d[:, 1] >= 0) & (keypoints_depth < keypoints_depth_proj + depth_diff_thres)
        if rgb is not None:
            canvas = (rgb.detach().cpu().numpy().reshape(H, W, 3) * 255).astype(np.uint8)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
        return draw_openpose_map(canvas, keypoints_2d.detach().cpu().numpy(), keypoints_mask.cpu().numpy())
    
    def get_num_subdiv(self, global_step=1e7):
        self.global_step = global_step
        if self.opt.tet_subdiv_steps is not None:
            num_subdiv = 0
            for step in self.opt.tet_subdiv_steps:
                if global_step >= step:
                    num_subdiv += 1
            return num_subdiv
        return self.opt.tet_num_subdiv


    def forward(self, rays_o, rays_d, mvp, h0, w0, smpl_v_T=None, light_d=None, ref_rgb=None, ambient_ratio=1.0, shading='albedo', return_loss=False, alpha_only=False, detach_geo=False, albedo_ref=False, poses=None, return_openpose_map=False, global_step=1e7, return_can_pos_map=False, mesh=None, can_pose=False):
        # mvp: [1, 4, 4]
        mvp = mvp.squeeze()
        device = mvp.device

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        # do super-sampling
        if self.opt.ssaa > 1:
            h = int(h0 * self.opt.ssaa)
            w = int(w0 * self.opt.ssaa)
            if not self.opt.use_gl:
                h  = min(h, 2048)
                w  = min(w, 2048)
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w = h0, w0
            dirs = rays_d

        if self.opt.single_bg_color:
            dirs = torch.ones_like(dirs)

        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        dirs[..., 0] = -dirs[..., 0]
        dirs[..., 2] = -dirs[..., 2]

        # mix background color
        if self.opt.different_bg and shading == 'textureless':
            bg_color = torch.sigmoid(self.textureless_bg_net(self.encoder_bg(dirs))).view(h, w, 3)
        elif self.opt.different_bg and shading == 'normal':
            bg_color = torch.sigmoid(self.normal_bg_net(self.encoder_bg(dirs))).view(h, w, 3)
        else:
            bg_color = torch.sigmoid(self.bg_net(self.encoder_bg(dirs))).view(h, w, 3)

        results = {}
        geo_reg_loss = None
        if mesh is None:
            mesh, geo_reg_loss = self.get_mesh(return_loss=return_loss, detach_geo=detach_geo, global_step=global_step)

        results['mesh'] = mesh
        v = mesh.v  # [N, 3]
        f = mesh.f
        if can_pose:
            v, f = random.choice(list(zip(self.can_pose_vertices, self.can_pose_faces)))

        v_clip = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0),
                              torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]
        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, H, W, 1]
        alpha = mask.clone()
        if alpha_only:
            alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]
            if self.opt.ssaa > 1:
                alpha = scale_img_hwc(alpha, (h0, w0))
            return dict(alpha=alpha)
        # xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3]
        # xyzs = xyzs.view(-1, 3)
        # mask = (mask > 0).view(-1)
        # albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        # if mask.any():
        #     masked_albedo = torch.sigmoid(self.color_net(self.encoder(xyzs[mask].detach(), bound=1)))
        #     albedo[mask] = masked_albedo.float()
        # albedo = albedo.view(1, h, w, 3)cuda
        if not self.opt.use_can_pose_space:
            color_space_v, color_space_f = mesh.v, mesh.f
        else:
            color_space_v, color_space_f = self.can_pose_vertices[0], self.can_pose_faces[0]


        if shading != 'albedo' and light_d is None:  # random sample light_d if not provided
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = rays_o[0] + torch.randn(3, device=rays_o.device, dtype=torch.float)
            #light_d = random.choice(-rays_d.view(-1, 3))#(rays_o[0] + torch.randn(3, device=rays_o.device, dtype=torch.float))
            light_d = safe_normalize(light_d)
        if shading in ['normal', 'textureless']:
            color = self.get_color_from_mesh(mesh, rast, light_d, ambient_ratio, shading, poses=poses)
        elif self.opt.use_texture_2d:
            color = self.get_color_from_2d_texture(rast, rast_db, mesh, rays_o, light_d, ambient_ratio, shading)
        elif self.opt.use_vertex_tex:
            color = self.get_color_from_vertex_texture(rast, rast_db, color_space_f, light_d, ambient_ratio, shading)
        else:
            if self.opt.detach_tex:
                color = self.get_color_from_3d_texture(rast.detach(), rast_db, color_space_v, color_space_f, mesh.vn.detach(), light_d, ambient_ratio, shading)
            else:
                color = self.get_color_from_3d_texture(rast, rast_db, color_space_v, color_space_f, mesh.vn, light_d, ambient_ratio, shading)

        color = dr.antialias(color, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]
        # color = color.squeeze(0).clamp(0, 1)
        # alpha = alpha.squeeze(0).clamp(0, 1)
        depth = rast[0, :, :, [2]]  # [H, W]

        color = color * alpha + (1 - alpha) * bg_color

        # ssaa

        if albedo_ref and not (self.opt.use_vertex_tex) and (not self.opt.use_texture_2d):
            with torch.no_grad():
                albedo = self.get_color_from_3d_texture(rast.detach(), rast_db.detach(), color_space_v.detach(), color_space_f, None, light_d, 1.0, 'albedo')
                albedo = dr.antialias(albedo, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1)  # [H, W, 3]
                albedo = albedo * alpha + (1 - alpha) * bg_color
            if self.opt.ssaa > 1:
                albedo = scale_img_hwc(albedo, (h0, w0))
            results['albedo_ref'] = albedo

        if self.opt.ssaa > 1:
            color = scale_img_hwc(color, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            bg_color = scale_img_hwc(bg_color, (h0, w0))

        results['depth'] = depth
        results['image'] = color
        results['alpha'] = alpha
        results['bg_color'] = bg_color
        if geo_reg_loss is not None:
            results['geo_reg_loss'] = geo_reg_loss
        
        if return_openpose_map:
            keypoints_2d = torch.matmul(F.pad(self.keypoints, pad=(0, 1), mode='constant', value=1.0),
                              torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]
            results['openpose_map'] = depth.new_tensor(self.get_openpose_map(keypoints_2d, depth, color if self.opt.test else None)) / 255
            # results['image'] = torch.flip(results['image'], dims=[-2])
            # results['openpose_map'] = torch.flip(results['openpose_map'], dims=[-2])
        if return_can_pos_map:
            results['can_pos_map'] = self.get_can_pos_map(rast.detach(), rast_db.detach(), mesh.f)
            if self.opt.ssaa > 1:
                results['can_pos_map'] = scale_img_hwc(results['can_pos_map'], (h0, w0))

        
        return results