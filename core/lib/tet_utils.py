import pyvista as pv
import pymeshlab
import tetgen
import os.path as osp
import os
import numpy as np

import open3d as o3d
import torch

def build_tet_grid(mesh, cfg):
    assert cfg.data.last_model.split('.')[-1] == 'obj'
    tet_dir = osp.join(cfg.workspace, 'tet')
    os.makedirs(tet_dir, exist_ok=True)
    save_path = osp.join(tet_dir, 'tet_grid.npz')
    if osp.exists(save_path):
        print('Loading exist tet grids from {}'.format(save_path))
        tets = np.load(save_path)
        vertices = tets['vertices']
        indices = tets['indices']
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        return vertices, indices
    print('Building tet grids...')
    tet_flag = False
    tet_shell_offset = cfg.model.tet_shell_offset
    while (not tet_flag) and tet_shell_offset > cfg.model.tet_shell_offset / 16:
        # try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.v.cpu().numpy(), mesh.f.cpu().numpy()))
        ms.generate_resampled_uniform_mesh(offset=pymeshlab.AbsoluteValue(tet_shell_offset))
        ms.save_current_mesh(osp.join(tet_dir, 'dilated_mesh.obj'))
        mesh = pv.read(osp.join(tet_dir, 'dilated_mesh.obj'))
        downsampled_mesh = mesh.decimate(cfg.model.tet_shell_decimate)
        tet = tetgen.TetGen(downsampled_mesh)
        tet.make_manifold(verbose=True)
        vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=cfg.model.tet_grid_volume, 
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
        shell = tet.grid.extract_surface()
        shell.save(osp.join(tet_dir, 'shell_surface.ply'))
        np.savez(save_path, vertices=vertices, indices=indices)
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        tet_flag = True
        # except:
        #     tet_shell_offset /= 2
    assert tet_flag, "Failed to initialize tetrahedra grid!"
    return vertices, indices


def simplify_smpl(mesh_v, smpl_faces):
    mesh = o3d.geometry.TriangleMesh()
    np_vertices = mesh_v.cpu().numpy()
    np_triangles = smpl_faces.cpu().numpy()

    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=1500)

    vertices_smp = np.array(mesh_smp.vertices)
    faces_smp = np.array(mesh_smp.triangles)

    return vertices_smp, faces_smp


def build_tet_grid_new(mesh, opt, smpl_rest_v=None, smpl_faces=None):
    assert opt.last_model.split('.')[-1] == 'obj'
    tet_dir = osp.join(opt.workspace, 'tet')
    os.makedirs(tet_dir, exist_ok=True)
    save_path = osp.join(tet_dir, 'tet_grid.npz')

    if osp.exists(save_path): # only to save some time
        print('Loading exist tet grids from {}'.format(save_path))
        tets = np.load(save_path)
        vertices = tets['vertices']
        indices = tets['indices']
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        return vertices, indices

    print('Building tet grids...')
    tet_flag = False
    tet_shell_offset = opt.tet_shell_offset
    print('tet_shell_offset', tet_flag, tet_shell_offset)

    import trimesh
    import pytorch3d

    mesh_v, mesh_f = simplify_smpl(mesh.v, smpl_faces)
    # print('mesh_v, mesh_f', mesh_v.shape, mesh_f.shape)
    mesh_v = torch.from_numpy(mesh_v).float().cuda()
    mesh_f = torch.from_numpy(mesh_f).long().cuda()

    # p3d_mesh = pytorch3d.structures.Meshes(verts=[mesh.v], faces=[smpl_faces])
    p3d_mesh = pytorch3d.structures.Meshes(verts=[mesh_v], faces=[mesh_f])
    verts_normal = p3d_mesh.verts_normals_list()[0]
    mesh_v = mesh_v + verts_normal.detach() * 0.1

    pred_mesh_rest2 = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
    pred_mesh_rest2.export(osp.join(tet_dir, 'dilated_mesh.obj'))

    # assert False
    print('start to build tet grid')
    
    if True:
        ms = pymeshlab.MeshSet()
        # ms.add_mesh(pymeshlab.Mesh(vertices_smp, faces_smp))
        ms.add_mesh(pymeshlab.Mesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy()))

        mesh = pv.read(osp.join(tet_dir, 'dilated_mesh.obj'))
        downsampled_mesh = mesh.decimate(opt.tet_shell_decimate)
        tet = tetgen.TetGen(downsampled_mesh)
        tet.make_manifold(verbose=True)
        print('opt.tet_grid_volume', opt.tet_grid_volume)
        vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=opt.tet_grid_volume, 
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
        shell = tet.grid.extract_surface()
        shell.save(osp.join(tet_dir, 'shell_surface.ply'))
        np.savez(save_path, vertices=vertices, indices=indices)
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        tet_flag = True

    assert tet_flag, "Failed to initialize tetrahedra grid!"
    return vertices, indices