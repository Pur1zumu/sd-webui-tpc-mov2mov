import os.path
import time

import cv2
from PIL import Image, ImageOps

import modules.images
from modules import shared, sd_samplers, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
import modules.scripts as scripts
from scripts.m2m_util import get_mov_all_images, images_to_video
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from modules.ui import plaintext_to_html
from scripts.m2m_modnet import get_model, infer, infer2
from scripts.kv_mem import kv_mem, split_cross_attention_forward
import ldm.modules.attention


def process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, w, h, generate_mov_mode, extract_characters,
                    merge_background,
                    modnet_model,
                    args, max_kvmem_size, kvmem_retain_first):
    processing.fix_seed(p)

    if not kv_mem.func_hacked:
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        kv_mem.func_hacked = True
    kv_mem.reset()
    kv_mem.max_mem_size = max_kvmem_size
    kv_mem.retain_first = kvmem_retain_first

    images = get_mov_all_images(mov_file, movie_frames)
    if not images:
        print('Failed to parse the video, please check')
        return

    print(f'The video conversion is completed, images:{len(images)}')
    if max_frames == -1 or max_frames > len(images):
        max_frames = len(images)

    max_frames = int(max_frames)

    p.do_not_save_grid = True
    state.job_count = max_frames  # * p.n_iter
    generate_images = []
    for i, image in enumerate(images):
        if i >= max_frames:
            break

        kv_mem.iter_reset(i)
        kv_mem.clean_unused_mem()

        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        modnet_network = None
        # 处理modnet

        # 存一张底图

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        img = ImageOps.exif_transpose(img)
        if extract_characters:
            print(f'loading modnet model: {modnet_model}')
            modnet_network = get_model(modnet_model)
            print(f'Loading modnet model completed')
            img, _ = infer2(modnet_network, img)

        p.init_images = [img] * p.batch_size
        proc = scripts.scripts_img2img.run(p, *args)
        if proc is None:
            print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            # 只取第一张

            gen_image = processed.images[0]

            if extract_characters and merge_background:
                backup = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
                backup = modules.images.resize_image(resize_mode, backup, w, h)
                _, mask = infer2(modnet_network, backup)
                gen_image = Image.composite(gen_image, backup, mask)

            generate_images.append(gen_image)


    if not os.path.exists(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir)):
        os.makedirs(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), exist_ok=True)

    if generate_mov_mode == 0:
        r_f = '.mp4'
        mode = 'mp4v'
    elif generate_mov_mode == 1:
        r_f = '.mp4'
        mode = 'avc1'
    elif generate_mov_mode == 2:
        r_f = '.avi'
        mode = 'XVID'

    print(f'Start generating {r_f} file')

    video = images_to_video(generate_images, movie_frames, mode, w, h,
                            os.path.join(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), str(int(time.time())) + r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video


def mov2mov(id_task: str,
            prompt,
            negative_prompt,
            mov_file,
            steps,
            sampler_index,
            restore_faces,
            tiling,
            extract_characters,
            merge_background,
            modnet_model,
            # fixed_seed,
            generate_mov_mode,
            noise_multiplier,
            # color_correction,
            cfg_scale,
            image_cfg_scale,
            denoising_strength,
            movie_frames,
            max_kvmem_size, kvmem_retain_first,
            max_frames,
            seed,
            subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,
            height,
            width,
            resize_mode,
            override_settings_text, *args):
    if not mov_file:
        raise Exception('Error！ Please add a video file!')

    override_settings = create_override_settings_dict(override_settings_text)
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    mask_blur = 4
    inpainting_fill = 1
    inpaint_full_res = False
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples= shared.opts.data.get("mov2mov_outpath_samples", mov2mov_outpath_samples),
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[None],
        mask=None,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=noise_multiplier
    )

    p.scripts = scripts.scripts_img2img
    p.script_args = args

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nmov2mov: {prompt}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    print(f'\nStart parsing the number of mov frames')

    generate_video = process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, width, height,
                                     generate_mov_mode,
                                     extract_characters, merge_background, modnet_model, args, max_kvmem_size, kvmem_retain_first)
    processed = Processed(p, [], p.seed, "")
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generate_video, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments)
