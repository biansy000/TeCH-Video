#import nvdiffrast.torch as dr
import torch
import argparse

from dream.provider import ViewDataset
from dream.utils import *
from dream.renderer import Renderer
from dream.gui import GUI


#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--da_pose_mesh', type=str, help="mesh template, must be obj format")
    parser.add_argument(
        '--last-model', type=str, required=True, help='trained model "*.pth" or mesh ".obj" by last stage')
    parser.add_argument(
        '--smpl-model', type=str, help='SMPL mesh ".obj"')
    parser.add_argument(
        '--last-ref-model', type=str, help='trained model "*.pth" or mesh ".obj" by last stage')
    parser.add_argument(
        '--keypoints-path', type=str, help='".npy" smplx keypoints annotation')
    parser.add_argument('--use-texture-2d', action='store_true')
    parser.add_argument('--use-vertex-tex', action='store_true')
    parser.add_argument('--init-texture-3d', action='store_true')
    parser.add_argument('--init-mesh', action='store_false', default=True)
    parser.add_argument('--mesh-resolution', type=int, default=128, help="The resolution of mesh")
    parser.add_argument('--mesh-scale', type=float, default=1.0, help="The scaling ratio of mesh")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--text_geo', default=None, help="text prompt for geometry")
    parser.add_argument('--text_head', default=None, help="text prompt")
    parser.add_argument('--text-extra', default='', help="text prompt")
    parser.add_argument('--normal-text', default=None, help="normal training mode text prompt")
    parser.add_argument('--textureless-text', default=None, help="textureless training mode text prompt")
    parser.add_argument('--normal-text-extra', default='', help="normal training mode text prompt")
    parser.add_argument('--textureless-text-extra', default='', help="textureless training mode text prompt")
    parser.add_argument('--img', default=None, help="reconstruct image")
    parser.add_argument('--loss-mask', default=None, type=str, help="reconstruct loss mask")
    parser.add_argument('--occ-mask', default=None, type=str, help="reconstruct loss mask")
    parser.add_argument('--seg-mask', default=None, type=str, help="reconstruct loss mask")
    parser.add_argument('--loss-mask-erosion', default=None, type=int, help="reconstruct mask erosion size")
    parser.add_argument('--normal-img', default=None, help="reconstruct normal estimation image")
    parser.add_argument('--back-normal-img', default=None, help="reconstruct back normal estimation image")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--negative_normal', default='', type=str, help="negative text prompt")
    parser.add_argument('--negative_textureless', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--not_test_video', action='store_true', default=False, help="test mode")
    parser.add_argument('--use_gl', action='store_true', default=False, help="test mode")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh")
    parser.add_argument('--save_uv', action='store_true', help="export an obj mesh with uv texture")
    parser.add_argument('--color-correction', action='store_true', help="export an obj mesh with texture & color correction")

    ### training options
    parser.add_argument('--lock_geo', action='store_true', help="fix geometry")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument(
        '--albedo', action='store_true', help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--albedo_ratio', type=float, default=0.2, help="training iters that only use albedo shading")
    parser.add_argument('--decay-norm-recon-cosine-cycle', type=int, default=None, help="cycle of cosine decay the normal reconstuction loss weight")
    parser.add_argument('--decay-norm-recon-cosine-max-iter', type=int, default=20000, help="max iteration of cosine decay the normal reconstuction loss weight")
    parser.add_argument('--decay-norm-recon-iter', type=int, nargs='*', default=[], help="after this iter, decay the normal reconstuction loss weight")
    parser.add_argument('--decay-norm-recon-ratio', type=float, nargs='*', default=[], help="the ratio to decay the normal reconstuction loss weight")
    parser.add_argument('--back-norm-recon-iter', type=int, default=1000000, help="after this iter, decay the normal reconstuction loss weight")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument(
        '--uniform_sphere_rate',
        type=float,
        default=0.5,
        help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument(
        '--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--head_hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--hf_key_lora', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--lora', type=str, default=None, help="load lora weights")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--w', type=int, default=512, help="render width in training")
    parser.add_argument('--h', type=int, default=512, help="render height in training")
    parser.add_argument('--anneal_tex_reso', action='store_true', help="increase h/w gradually for smoother texture")
    parser.add_argument('--init_empty_tex', action='store_true', help="always initialize an empty texture")

    parser.add_argument('--remesh', action='store_true', help="track face error and do subdivision")
    parser.add_argument("--remesh_steps_ratio", type=float, action="append", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--remesh_size', type=float, default=0.02, help="remesh trig length")

    parser.add_argument('--lambda_offsets', type=float, default=0., help="loss scale")
    parser.add_argument('--lambda_normal_offsets', type=float, default=0., help="loss scale")
    parser.add_argument('--lambda_lap', type=float, default=0., help="loss scale")
    parser.add_argument('--lambda_normal', type=float, default=0., help="loss scale")
    parser.add_argument('--lambda_edgelen', type=float, default=0., help="loss scale")
    parser.add_argument('--lambda_recon', type=float, default=0., help="reconstruction loss weight")
    parser.add_argument('--lambda_normal_recon', type=float, default=0., help="normal reconstruction loss weight")
    parser.add_argument('--lambda_gradient', type=float, default=0., help="grad error loss weight")

    ### dataset options
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")
    parser.add_argument(
        '--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument(
        '--theta_range', type=float, nargs='*', default=[0, 120], help="training camera theta range")
    parser.add_argument(
        '--phi_diff', type=float, default=0, help="training camera phi diff")
    parser.add_argument('--height_range', type=float, nargs=2, default=[0., 0.], help="training camera height range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument(
        '--dir_text',
        action='store_true',
        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument(
        '--angle_front',
        type=float,
        default=60,
        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--phi-range', type=float, nargs=2, default=[0., 360.], help="canonical poses files")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument(
        '--light_theta',
        type=float,
        default=60,
        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    parser.add_argument('--guidance_scale', type=float, default=100, help="prompt guidance scale for SDS loss")
    parser.add_argument('--normal_guidance_scale', type=float, default=100, help="prompt guidance scale for SDS loss")
    parser.add_argument('--step_range', type=float, nargs=2, default=[0.02, 0.98], help="step sample range for SDS loss")
    parser.add_argument('--max_step_percent_annealed', type=float, default=0.5, help="step sample range for SDS loss")
    parser.add_argument('--anneal_start_step', type=int, default=100000000, help="step sample range for SDS loss")
    parser.add_argument('--mesh_dataset', type=str, default="phorhum", help="mesh dataset")
    parser.add_argument('--albedo_res', type=int, default=2048, help="albedo uv map resolution")

    parser.add_argument('--use-dmtet-new', action='store_true', help='use a SDF network and outer shell')
    parser.add_argument('--use-dmtet-posed', action='store_true', help='use a SDF network and outer shell, posed one')
    parser.add_argument('--progressive-geo', action='store_true', help='progressive_geo')
    parser.add_argument('--use-pose-control', action='store_true', help='use a pose control network')
    parser.add_argument('--use-explicit-tet', action='store_true', help='use a SDF network and outer shell')
    parser.add_argument('--tet-shell-offset', type=float, default=0.1, help="tetrahedra shell outer offset")
    parser.add_argument('--tet-shell-decimate', type=float, default=0.9, help="tetrahedra shell decimate ratio")
    parser.add_argument('--tet-grid-scale', type=float, default=1e-4, help="tetrahedra shell grid scale")
    parser.add_argument('--tet-grid-volume', type=float, default=5e-7, help="tetrahedra shell decimate ratio")
    parser.add_argument('--tet-num-subdiv', type=int, default=0, help="tetrahedra subdivision times")
    parser.add_argument('--tet-subdiv-steps', type=int, nargs='*', default=None, help="tetrahedra subdivision times / step")
    parser.add_argument('--dmtet-lr', type=float, default=0.1, help="dmtet network learning rate")
    parser.add_argument('--dmtet-network', type=str, default='mlp', help="dmtet network (mlp or hash)")
    parser.add_argument('--init-mesh-padding', type=float, default=0., help="init mesh outer padding")
    
    parser.add_argument('--normal-train', type=float, default=-1, help="train ratio in normal mode")
    parser.add_argument('--train-geometry-step', type=int, nargs=2, default=[0, 10000000], help="step range to train ")
    parser.add_argument('--textureless-train', type=float, default=-1, help="train in textureless mode")
    parser.add_argument('--decouple-recon-loss', action='store_true', help="decouple geometry and albedo in geometry loss")
    parser.add_argument('--use-lap-loss-tt', action='store_true', help="use another implementation of lap loss")
    parser.add_argument('--use-sil-loss', action='store_true', help="use silhouette loss")
    parser.add_argument('--different_bg', action='store_true', help="use different bg_net for normal textureless and color")
    parser.add_argument('--single-bg-color', action='store_true', help="learn single background color")
    parser.add_argument('--detach-tex', action='store_true', help="always decouple texture and geometry")
    parser.add_argument('--controlnet-guidance-geometry', type=str, default=None, help="controlnet guidance for geometry")
    parser.add_argument('--controlnet-openpose-guidance', action='store_true', help="controlnet openpose guidance for geometry and texture")
    parser.add_argument('--controlnet', type=str, default=None, help="controlnet model")
    parser.add_argument('--controlnet-conditioning-scale', type=float, default=1.0, help="controlnet conditioning scale")
    parser.add_argument('--controlnet_guide_inputview', action='store_true', help="use controlnet guidance on input view geometry")
    parser.add_argument('--face-ratio', type=float, default=0., help="face sampling ratio")
    parser.add_argument('--face-height-range', type=float, nargs=2, default=[0.4, 0.4], help="face height")
    parser.add_argument('--face-radius-range', type=float, nargs=2, default=[0.3, 0.4], help="face radius range")
    parser.add_argument('--face-phi-diff', type=float, default=0., help="face phi diff")
    parser.add_argument('--face-phi-range', type=float, nargs=2, default=[0.,0.], help="face phi range")
    parser.add_argument('--face-theta-range', type=float, nargs=2, default=[0.,0.], help="face theta range")
    parser.add_argument('--can-pose-folder', type=str, default=None, help="canonical poses files")
    parser.add_argument('--can-pose-ratio', type=float, default=0.5, help="canonical poses sampling ratio")
    parser.add_argument('--use_can_pose_space', action='store_true', help="canonical poses color space")

    parser.add_argument('--geo-hash-max-res', type=int, default=1024)
    parser.add_argument('--geo-hash-num-levels', type=int, default=16)
    parser.add_argument('--rgb-hash-max-res', type=int, default=2048)
    parser.add_argument('--rgb-hash-num-levels', type=int, default=16)

    parser.add_argument('--render-relative-normal', action='store_true', help="render relative normal map")
    parser.add_argument('--write-image', action='store_true', help='render images as output')

    # vds configs
    parser.add_argument('--lora-training', action='store_true', help = 'train lora model')
    parser.add_argument('--half-precision-weights', action='store_true', help = 'half precision weights for lora model')
    parser.add_argument('--enable_attention_slicing', action='store_true', help = 'attention slicing for sd model')
    parser.add_argument('--lora-n-timestamp-samples', default=1, type=int)
    parser.add_argument('--guidance_scale_lora', type=float, default=1.0, help="prompt guidance scale for VDS lora loss")
    parser.add_argument('--camera_condition_type', type=str, default='extrinsics', help="VDS camera condition type")
    parser.add_argument('--lambda_lora', type=float, default=1., help="lora training loss weight")
    parser.add_argument('--lora_lr', type=float, default=1e-3, help="max learning rate")

    parser.add_argument('--ref-image-step', type=int, default=0, help="step range using image diffusion reference")

    parser.add_argument('--clip_step_range', type=float, default=0., help="step range using clip sds loss")
    parser.add_argument('--clip_guidance_scale', type=float, default=10., help="clip sds guidance scale")
    parser.add_argument('--lambda_clip_img_loss', type=float, default=0., help="clip sds image loss")
    parser.add_argument('--lambda_clip_text_loss', type=float, default=0., help="clip sds text loss")
    parser.add_argument('--lambda_color_chamfer', type=float, default=0., help="color chamfer distance loss")

    parser.add_argument('--single_directional_color_chamfer', action='store_true', help="color chamfer distance loss")
    parser.add_argument('--color_chamfer_step', type=int, default=0, help="use color chamfer loss after certain step")
    parser.add_argument('--can_pos_color_chamfer', action='store_true', help="canonical position color chamfer distance loss")
    parser.add_argument('--color_chamfer_space', type=str, default='rgb', help="color space for color chamfer distance loss")

    parser.add_argument('--incomplete-input', action='store_true')

    parser.add_argument('--concat_ref_image', type=float, default=0., help="concat reference image")
    parser.add_argument('--mixup_guidance_geometry', type=float, default=0., help="mixup_guidance_geometry")

    parser.add_argument('--train_both', action='store_true', help="train both geometry and texture in each iter")
    parser.add_argument('--concat_texture_geometry', action='store_true', help="concat both geometry and texture for training")

    parser.add_argument('--progressive_phi_step', type=int, default=None, help="progressive phi sampling max step")
    parser.add_argument('--start_phi_range', type=float, default=None, help="progressive phi sampling start range")

    parser.add_argument('--profile', action='store_true')





    opt = parser.parse_args()

    # opt.fp16 = True # TODO: lead to mysterious NaNs in backward ???

    opt.dir_text = True

    if opt.albedo:
        opt.albedo_iters = opt.iters

    assert not opt.remesh, "not correctly implemented now..."

    if opt.remesh:
        opt.remesh_steps = [int(round(x * opt.iters)) for x in opt.remesh_steps_ratio]

    print(opt)

    seed_everything(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if opt.use_dmtet_posed:
    if True:
        from src.models.smpl.PIXIE_SMPLX import SMPLX
        from src.models.smpl.smpl_config import cfg_model
        import trimesh

        keypoints_path = opt.keypoints_path
        smpl_info = np.load(keypoints_path, allow_pickle=True).item()

        smpl_scale = smpl_info["scale"]
        smpl_trans = smpl_info["transl"]
        disp_coord = torch.tensor([1.0, -1.0, -1.0], device=smpl_scale.device)
        smpl_model = SMPLX(cfg_model)
        smpl_v, smpl_v_rest, _, smpl_joints, joints_rest, T_smpl = get_smpl_out(smpl_model, smpl_info)

        # print('smpl_scale', smpl_scale)
        # assert False

        # smpl_v_rest, smpl_lm_rest, smpl_joints_rest, T_rest = get_smpl_out_zero_pose(smpl_model, smpl_info)
        smpl_faces = smpl_model.faces_tensor.unsqueeze(0)

        # smpl_v_rest += smpl_trans
        # smpl_v_rest *= smpl_scale
        # smpl_v_rest = smpl_v_rest * disp_coord
        # print('T_smpl', T_smpl[:, 19])
        # print('smpl_v_rest', smpl_v_rest.max(), smpl_v_rest.min())
        # assert False
        T_smpl[:, :, :3, 3] = T_smpl[:, :, :3, 3] + smpl_trans
        T_smpl[:, :, :3, :] = T_smpl[:, :, :3, :] * smpl_scale
        T_smpl[:, :, :3, :] = T_smpl[:, :, :3, :] * disp_coord.unsqueeze(-1)

        T_rest = torch.eye(4, device=T_smpl.device).reshape(1, 1, 4, 4)
        T_rest = T_rest.repeat(T_smpl.shape[0], T_smpl.shape[1], 1, 1)
        T_rest[:, :, :3, 3] = T_rest[:, :, :3, 3] + smpl_trans
        T_rest[:, :, :3, :] = T_rest[:, :, :3, :] * smpl_scale
        # T_rest[:, :, :3, :] = T_rest[:, :, :3, :] * disp_coord.unsqueeze(-1)

        rest_smpl_obj = trimesh.Trimesh(
            smpl_v_rest[0],
            smpl_model.faces_tensor[:, [0, 2, 1]],
            process=False,
            maintains_order=True,
        )

        smpl_obj_path = opt.last_model.replace('.obj', '_rest.obj')
        rest_smpl_obj.export(smpl_obj_path)

        smpl_v_rest = smpl_v_rest.detach().to(device)
        smpl_faces = smpl_faces.detach().to(device)
        # smpl_v = smpl_v.detach().to(device)
        T_rest = T_rest.detach().to(device)
        T_smpl = T_smpl.detach().to(device)

        # print('T_smpl', T_smpl[:, 5])
        # assert False
        # pred_mesh = trimesh.Trimesh(smpl_v_rest[0].cpu().numpy(), smpl_faces[0].cpu().numpy())
        # pred_mesh.export('../exp/smpl_try_rest.obj')

        # batch_size = 1
        # homogen_coord = torch.ones([smpl_v_rest.shape[0], smpl_v_rest.shape[1], 1], device=device)
        # v_posed_homo = torch.cat([smpl_v_rest, homogen_coord], dim=2)
        # smpl_v = torch.matmul(T_smpl, torch.unsqueeze(v_posed_homo, dim=-1))[:, :, :3, 0]
        # print('smpl_v', smpl_v.shape, smpl_faces.shape)
        # pred_mesh = trimesh.Trimesh(smpl_v[0].cpu().numpy(), smpl_faces[0].cpu().numpy())
        # pred_mesh.export('../exp/smpl_try.obj')
        # assert False
        # print(smpl_faces.shape, smpl_v_rest.shape, smpl_lm_rest.shape, smpl_joints_rest.shape, T_smpl.shape)
        # torch.Size([1, 20908, 3]) torch.Size([1, 10475, 3]) torch.Size([1, 68, 3]) torch.Size([1, 145, 3]) torch.Size([1, 10475, 4, 4])
        # assert False
        smpl_dict = {

        }
    else:
        smpl_v_rest = None
        smpl_faces = None
        smpl_v= None
        T_rest = None
        T_smpl = None

    print('building renderer')
    model = Renderer(opt, smpl_v_rest=smpl_v_rest, smpl_faces=smpl_faces, rest_T=T_rest, smpl_v_T=T_smpl, joints_rest=joints_rest)
    if model.keypoints is not None:
        if len(model.keypoints[0]) == 1:
            opt.head_location = model.keypoints[0][0].cpu().numpy()
        else:
            opt.head_location = model.keypoints[0][15].cpu().numpy()
    else:
        opt.head_location = np.array([0., 0.4, 0.], dtype=np.float32)
    opt.canpose_head_location = np.array([0., 0.4, 0.], dtype=np.float32)
    print('head_location', opt.head_location)
    print('renderer ok!')

    if opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer(
            'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt, pretrained=opt.pretrained)

        if opt.gui:
            gui = GUI(opt, trainer)
            gui.render()

        else:
            if not opt.not_test_video:
                test_loader = ViewDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100, render_head=True).dataloader()
                trainer.test(test_loader, write_image=opt.write_image)
                if opt.can_pose_folder is not None:
                    trainer.test(test_loader, write_image=opt.write_image, can_pose=True)  
            if opt.save_mesh:
                trainer.save_mesh()


    else:

        train_loader = ViewDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
        params_list = list()
        if opt.guidance == 'stable-diffusion':
            from sd import StableDiffusion
            guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, opt.step_range, controlnet=opt.controlnet, lora=opt.lora, cfg=opt, head_hf_key=opt.head_hf_key)
            for p in guidance.parameters():
                p.requires_grad = False
        elif opt.guidance == 'stable-diffusion-vds':
            from sd_vds import StableDiffusionVDS
            guidance = StableDiffusionVDS(device, opt.sd_version, opt.hf_key, opt.hf_key_lora, opt.step_range, controlnet=opt.controlnet, cfg=opt)
            params_list.extend(guidance.get_params(opt.lora_lr))
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')
        
        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            params_list.extend(model.get_params(5 * opt.lr))
            optimizer = lambda model: Adan(
                params_list, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            params_list.extend(model.get_params(opt.lr))
            optimizer = lambda model: torch.optim.Adam(params_list, betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1**min(iter / opt.iters, 1))


        trainer = Trainer(
            'df',
            opt,
            model,
            guidance,
            device=device,
            workspace=opt.workspace,
            optimizer=optimizer,
            ema_decay=None,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            use_checkpoint=opt.ckpt,
            eval_interval=opt.eval_interval,
            scheduler_update_every_step=True, 
            pretrained=opt.pretrained,
            )

        if opt.gui:
            trainer.train_loader = train_loader  # attach dataloader to trainer

            gui = GUI(opt, trainer)
            gui.render()

        else:
            valid_loader = ViewDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            if opt.profile:
                import cProfile
                with cProfile.Profile() as pr:
                    trainer.train(train_loader, valid_loader, max_epoch)        
                    pr.dump_stats(os.path.join(opt.workspace, 'profile.dmp'))
                    pr.print_stats()
            else:
                trainer.train(train_loader, valid_loader, max_epoch)

            test_loader = ViewDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100, render_head=True).dataloader()
            trainer.test(test_loader, write_image=opt.write_image)

            if opt.save_mesh:
                trainer.save_mesh()