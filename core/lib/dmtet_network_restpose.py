import torch
import torch.nn as nn
import kaolin as kal
from tqdm import tqdm
import random
import trimesh
from .encodings import HashEncoding
from src.models.mesh_utils import query_barycentric_weights, sample_sdf_near_surface_improved
import pymeshfix

# MLP + Positional Encoding
class Decoder(torch.nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 4, hidden = 8, multires = 5):
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter, device='cuda', axis_scale=1.):
        print ("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            p = torch.rand((1024,3), device=device) - 0.5
            p = p / axis_scale
            ref_value  = torch.sqrt((p**2).sum(-1)) - 0.3
            output = self(p)
            loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())


# Positional Encoding from https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it):
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    laplacian_weight = 0.1
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer


class HashDecoder(nn.Module):
    def __init__(self, input_dims = 3, internal_dims = 32, output_dims = 4, hidden = 2, input_bounds=None, max_res=1024, num_levels=16) -> None:
        super().__init__()
        self.input_bounds = input_bounds
        self.embed_fn = HashEncoding(implementation='torch', max_res=max_res, num_levels=num_levels)
        input_dims = self.embed_fn.get_out_dim()
        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def gradient(self, p):
        p.requires_grad_(True)
        if self.input_bounds is not None:
            x = (p - self.input_bounds[0]) / (self.input_bounds[1] - self.input_bounds[0])
        else:
            x = p
        if self.embed_fn is not None:
            x = self.embed_fn(x)
        y = self.net(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=p,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def forward(self, p):
        if self.input_bounds is not None:
            p = (p - self.input_bounds[0]) / (self.input_bounds[1] - self.input_bounds[0])
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p)
        return out


###############################################################################
# Compact tet grid
###############################################################################

def compact_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Find surface tets
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)  # one value per tet, these are the surface tets

        valid_vtx = tet_fx4[valid_tets].reshape(-1)
        unique_vtx, idx_map = torch.unique(valid_vtx, dim=0, return_inverse=True)
        new_pos = pos_nx3[unique_vtx]
        new_sdf = sdf_n[unique_vtx]
        new_tets = idx_map.reshape(-1, 4)
        return new_pos, new_sdf, new_tets


###############################################################################
# Subdivide volume
###############################################################################

def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)
        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
    return torch.stack([a, b], -1)


def batch_subdivide_volume(tet_pos_bxnx3, tet_bxfx4):
    device = tet_pos_bxnx3.device
    # get new verts
    tet_fx4 = tet_bxfx4[0]
    edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
    all_edges = tet_fx4[:, edges].reshape(-1, 2)
    all_edges = sort_edges(all_edges)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
    idx_map = idx_map + tet_pos_bxnx3.shape[1]
    all_values = tet_pos_bxnx3
    mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
        all_values.shape[0], -1, 2,
        all_values.shape[-1]).mean(2)
    new_v = torch.cat([all_values, mid_points_pos], 1)

    # get new tets

    idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
    idx_ab = idx_map[0::6]
    idx_ac = idx_map[1::6]
    idx_ad = idx_map[2::6]
    idx_bc = idx_map[3::6]
    idx_bd = idx_map[4::6]
    idx_cd = idx_map[5::6]

    tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
    tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
    tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
    tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
    tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
    tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
    tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
    tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

    tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
    tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
    tet = tet_np.long().to(device)

    return new_v, tet


class DMTetMeshRestPose(nn.Module):
    def __init__(self, vertices: torch.Tensor, indices: torch.Tensor, device: str='cuda', grid_scale=1e-4, use_explicit=False, geo_network='mlp', hash_max_res=1024, hash_num_levels=16, num_subdiv=0) -> None:
        super().__init__()
        self.device = device
        self.tet_v = vertices.to(device)
        self.tet_ind = indices.to(device)
        self.use_explicit = use_explicit # geo: False
        if self.use_explicit:
            self.sdf = nn.Parameter(torch.zeros_like(self.tet_v[:, 0]), requires_grad=True)
            self.deform = nn.Parameter(torch.zeros_like(self.tet_v), requires_grad=True)
        elif geo_network == 'mlp':
            self.decoder = Decoder().to(device)
        elif geo_network == 'hash':
            pts_bounds = (self.tet_v.min(dim=0)[0], self.tet_v.max(dim=0)[0])
            self.decoder = HashDecoder(input_bounds=pts_bounds, max_res=hash_max_res, num_levels=hash_num_levels).to(device)
        self.grid_scale = grid_scale
        self.num_subdiv = num_subdiv

    def query_decoder(self, tet_v):
        if self.tet_v.shape[0] < 1000000:
            return self.decoder(tet_v)
        else:
            chunk_size = 1000000
            results = []
            for i in range((tet_v.shape[0] // chunk_size) + 1):
                if i*chunk_size < tet_v.shape[0]:
                    results.append(self.decoder(tet_v[i*chunk_size: (i+1)*chunk_size]))
            return torch.cat(results, dim=0)

    def get_mesh(self, smpl_v_rest, smpl_faces, rest_T, smpl_v_T, return_loss=False, num_subdiv=None):
        if num_subdiv is None:
            num_subdiv = self.num_subdiv
        if self.use_explicit:
            sdf = self.sdf * 1
            deform = self.deform * 1
        else:
            pred = self.query_decoder(self.tet_v)
            sdf, deform = pred[:,0], pred[:,1:]
        verts_deformed = self.tet_v + torch.tanh(deform) * self.grid_scale / 2 # constraint deformation to avoid flipping tets
        tet = self.tet_ind
        for i in range(num_subdiv):
            verts_deformed, _, tet = compact_tets(verts_deformed, sdf, tet)
            verts_deformed, tet = batch_subdivide_volume(verts_deformed.unsqueeze(0), tet.unsqueeze(0))
            verts_deformed = verts_deformed[0]
            tet = tet[0]
            pred = self.query_decoder(verts_deformed)
            sdf, _ = pred[:,0], pred[:,1:]
        mesh_verts, mesh_faces = kal.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tet, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
        
        # trimesh.Trimesh(mesh_verts[0].detach().cpu().numpy(), mesh_faces[0].cpu().numpy()).export('predicted.obj')
        # trimesh.Trimesh(smpl_v_rest[0].detach().cpu().numpy(), smpl_faces[0].cpu().numpy()).export('smpl_v_rest.obj')
        # if smpl_v_rest is None:
        #     return mesh_verts[0], mesh_faces[0], None
        
        mesh_verts = mesh_verts[0].unsqueeze(0)
        mesh_faces = mesh_faces[0].unsqueeze(0)

        # print('mesh_verts', mesh_verts.shape, mesh_verts.max(), mesh_verts.min())
        bc_weights, nearest_face, v2f_dis = query_barycentric_weights(
            mesh_verts, smpl_v_rest, smpl_faces
        )

        nearest_smpl_face_idx = smpl_faces[0][torch.tensor(nearest_face[0]).long()]
        # print('nearest_smpl_face_idx', smpl_v_T[0][nearest_smpl_face_idx].device, bc_weights.device)
        mesh_T = sum([
            smpl_v_T[0][nearest_smpl_face_idx][:, i] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)
        ]).unsqueeze(0)

        mesh_T_rest = sum([
            rest_T[0][nearest_smpl_face_idx][:, i] * bc_weights[0, :, i].reshape(-1, 1, 1) for i in range(3)
        ]).unsqueeze(0)

        # print('mesh_T', mesh_T[:, 10], smpl_v_T[0, 6])
        # assert False
        # print('bc_weights', bc_weights[0, :10], mesh_T[0, 39])

        mesh_homo = torch.cat(
            [mesh_verts, torch.ones_like(mesh_verts)[:, :, :1]], dim=-1)
        
        # print('mesh_T, mesh_homo', mesh_T.shape, mesh_homo.shape)
        mesh_verts_posed = torch.einsum('bkmn,bkn->bkm', mesh_T, mesh_homo)[:, :, :3]
        mesh_verts_rest = torch.einsum('bkmn,bkn->bkm', mesh_T_rest, mesh_homo)[:, :, :3]

        mesh_verts, mesh_faces, mesh_verts_posed, mesh_verts_rest = mesh_verts[0], mesh_faces[0], mesh_verts_posed[0], mesh_verts_rest[0]
        # print('return_loss', return_loss)
        
        if return_loss:
            gradients = self.decoder.gradient(self.tet_v + self.grid_scale * (torch.rand_like(self.tet_v) - 0.5))
            gradients_error = (torch.linalg.norm(gradients.reshape(1, -1, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
            
            gradients_error = gradients_error.mean() + torch.square(v2f_dis.norm(dim=-1)).mean() * 0.01
            # print('torch.square(v2f_dis.norm(dim=-1))', torch.square(v2f_dis.norm(dim=-1)).shape, (torch.linalg.norm(gradients.reshape(1, -1, 3), ord=2,
            #                                 dim=-1) - 1.0).shape)
            return mesh_verts_posed, mesh_verts_rest, mesh_faces, gradients_error
        else:
            return mesh_verts_posed, mesh_verts_rest, mesh_faces, None
    
    def init_mesh(self, mesh_v, mesh_f, smpl_v_rest, smpl_faces, rest_T, smpl_v_T, init_padding=0.):
        num_pts = self.tet_v.shape[0]
        vclean, fclean = pymeshfix.clean_from_arrays(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
        mesh = trimesh.Trimesh(vclean, fclean)
        mesh.export('init_gt_mesh.obj')

        # print('mesh_v', mesh_v.max(), mesh_v.min(), self.tet_v.max(), self.tet_v.min())
        # print('self.tet_v', self.tet_v.shape, self.tet_v.max(dim=0), self.tet_v.min(dim=0))
        '''
        self.tet_v torch.Size([693648, 3]) torch.return_types.max(
        values=tensor([0.5369, 0.4278, 0.1846], device='cuda:0'),
        indices=tensor([ 13,  11, 132], device='cuda:0')) torch.return_types.min(
        values=tensor([-0.5174, -0.6006, -0.1811], device='cuda:0'),
        indices=tensor([203, 229, 143], device='cuda:0'))'''

        import mesh_to_sdf
        sdf_tet = torch.tensor(mesh_to_sdf.mesh_to_sdf(mesh, self.tet_v.cpu().numpy()), dtype=torch.float32).to(self.device) - init_padding
        sdf_mesh_v, sdf_mesh_f = kal.ops.conversions.marching_tetrahedra(self.tet_v.unsqueeze(0), self.tet_ind, sdf_tet.unsqueeze(0))
        sdf_mesh_v, sdf_mesh_f = sdf_mesh_v[0], sdf_mesh_f[0]
        if self.use_explicit:
            self.sdf.data[...] = sdf_tet[...]
        else:
            trimesh.Trimesh(sdf_mesh_v.cpu().numpy(), sdf_mesh_f.cpu().numpy()).export('estimated.obj')
            # assert False
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
            batch_size = 300000
            # iter = 1
            iter = 1000
            points, sdf_gt = sample_sdf_near_surface_improved(mesh) 
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
            sdf_gt = torch.tensor(sdf_gt, dtype=torch.float32).to(self.device)
            points = torch.cat([points, self.tet_v], dim=0)
            sdf_gt = torch.cat([sdf_gt, sdf_tet], dim=0)
            num_pts = len(points)
            for i in tqdm(range(iter)):
                sampled_ind = random.sample(range(num_pts), min(batch_size, num_pts))
                p = points[sampled_ind]
                pred = self.decoder(p)
                sdf, deform = pred[:,0], pred[:,1:]

                if True:
                    gradients = self.decoder.gradient(p)
                    gradients_error = (torch.linalg.norm(gradients.reshape(1, -1, 3), ord=2,
                                                dim=-1) - 1.0) ** 2
                
                    gradients_error = gradients_error.mean() * 0.1
                else:
                    gradients_error = 0
            
                loss = nn.functional.mse_loss(sdf, sdf_gt[sampled_ind]) + gradients_error # + (deform ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("init sdf loss:", loss.item())
            torch.save(self.decoder.state_dict(), 'tet_params.pth')
            with torch.no_grad():
                mesh_v, mesh_rest, mesh_f, _ = self.get_mesh(smpl_v_rest, smpl_faces, rest_T, smpl_v_T, return_loss=False)
            pred_mesh = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
            print('fitted mesh with num_vertex {}, num_faces {}'.format(mesh_v.shape[0], mesh_f.shape[0]))
            pred_mesh.export('init_tmp.obj')

            pred_mesh_rest = trimesh.Trimesh(mesh_rest.cpu().numpy(), mesh_f.cpu().numpy())
            pred_mesh_rest.export('init_tmp_rest.obj')

            # assert False
