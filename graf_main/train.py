import argparse
import os
import gc
from os import path
import time
import copy
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Trainer, Evaluator
from graf.config import get_data, build_models, save_config, update_config, build_lr_scheduler
from graf.utils import count_trainable_parameters, get_nsamples, InfiniteSamplerWrapper
from graf.transforms import ImgToPatch

from GAN_stability.gan_training import utils
from GAN_stability.gan_training.train import update_average
from GAN_stability.gan_training.logger import Logger
from GAN_stability.gan_training.checkpoints import CheckpointIO
from GAN_stability.gan_training.distributions import get_ydist, get_zdist
from GAN_stability.gan_training.config import (
    load_config, build_optimizers,
)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')

    args, unknown = parser.parse_known_args() 
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    restart_every = config['training']['restart_every']
    fid_every = config['training']['fid_every']
    save_every = config['training']['save_every']
    backup_every = config['training']['backup_every']
    save_best = config['training']['save_best']
    assert save_best=='fid' or save_best=='kid', 'Invalid save best metric!'

    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save config file
    save_config(os.path.join(out_dir, 'config.yaml'), config)

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr, render_poses.shape)

    train_loader = iter(torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=False, pin_memory=True, sampler=InfiniteSamplerWrapper(train_dataset)
    ))

    val_dataset = train_dataset
    val_loader = train_loader
    hwfr_val = hwfr

    # Create models
    generator_1, discriminator_1 = build_models(config)
    generator_2, discriminator_2 = build_models(config)
    generator_3, discriminator_3 = build_models(config)
    
    print('Generator params: %d' % count_trainable_parameters(generator_1))
    print('Discriminator params: %d, channels: %d' % (count_trainable_parameters(discriminator_1),
                                                      discriminator_1.nc))
    print(generator_1.render_kwargs_train['network_fn'])
    print(discriminator_1)
    print('Generator params: %d' % count_trainable_parameters(generator_2))
    print('Discriminator params: %d, channels: %d' % (count_trainable_parameters(discriminator_2),
                                                      discriminator_2.nc))
    print(generator_2.render_kwargs_train['network_fn'])
    print(discriminator_2)
    
    print('Generator params: %d' % count_trainable_parameters(generator_3))
    print('Discriminator params: %d, channels: %d' % (count_trainable_parameters(discriminator_3),
                                                      discriminator_3.nc))
    print(generator_3.render_kwargs_train['network_fn'])
    print(discriminator_3)

    # Put models on gpu if needed
    generator_1 = generator_1.to(device)
    discriminator_1 = discriminator_1.to(device)
    generator_2 = generator_2.to(device)
    discriminator_2 = discriminator_2.to(device)
    generator_3 = generator_3.to(device)
    discriminator_3 = discriminator_3.to(device)

    g_optimizer_1, d_optimizer_1 = build_optimizers(
        generator_1, discriminator_1, config
    )
    g_optimizer_2, d_optimizer_2 = build_optimizers(
        generator_2, discriminator_2, config
    )
    g_optimizer_3, d_optimizer_3 = build_optimizers(
        generator_3, discriminator_3, config
    )

    # input transform
    img_to_patch = ImgToPatch(generator_1.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    ##checkpoint_io.register_modules(
        #discriminator=discriminator,
        #g_optimizer=g_optimizer,
       # d_optimizer=d_optimizer,
        #**generator.module_dict     # treat NeRF specially
    #)
    # Get model file
    model_file = config['training']['model_file']
    stats_file = 'stats_00049999.p'

    # Logger
    logger = Logger(
        log_dir=path.join(out_dir, 'logs'),
        img_dir=path.join(out_dir, 'imgs'),
        monitoring=config['training']['monitoring'],
        monitoring_dir=path.join(out_dir, 'monitoring')
    )

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Save for tests
    n_test_samples_with_same_shape_code = config['training']['n_test_samples_with_same_shape_code']
    ntest = batch_size
    x_real = get_nsamples(train_loader, ntest)
    ytest = torch.zeros(ntest).to(device)
    ztest = zdist.sample((ntest,)).to(device)
    ptest = torch.stack([generator_1.sample_pose() for i in range(ntest)]).to(device)
    if n_test_samples_with_same_shape_code > 0:
        ntest *= n_test_samples_with_same_shape_code
        ytest = ytest.repeat(n_test_samples_with_same_shape_code)
        ptest = ptest.unsqueeze_(1).expand(-1, n_test_samples_with_same_shape_code, -1, -1).flatten(0, 1)       # (ntest x n_same_shape) x 3 x 4

        zdim_shape = config['z_dist']['dim'] - config['z_dist']['dim_appearance']
        # repeat shape code
        zshape = ztest[:, :zdim_shape].unsqueeze(1).expand(-1, n_test_samples_with_same_shape_code, -1).flatten(0, 1)
        zappearance = zdist.sample((ntest,))[:, zdim_shape:]
        ztest = torch.cat([zshape, zappearance], dim=1)

    utils.save_images(x_real, path.join(out_dir, 'real.png'))

    # Test generator
    if config['training']['take_model_average']:
        generator_test = copy.deepcopy(generator_1)
        # we have to change the pointers of the parameter function in nerf manually
        generator_test.parameters = lambda: generator_test._parameters
        generator_test.named_parameters = lambda: generator_test._named_parameters
        checkpoint_io.register_modules(**{k+'_test': v for k, v in generator_test.module_dict.items()})
    else:
        generator_test = generator_1

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device, inception_nsamples=33)

    # Initialize fid+kid evaluator
    if fid_every > 0:
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file,
                                                   act_cache_file=kid_cache_file)

    # Train
    tstart = t0 = time.time()

    # Load checkpoint if it exists
    try:
        load_dict = checkpoint_io.load(model_file)
    except FileNotFoundError:
        it = epoch_idx = -1
        fid_best = float('inf')
        kid_best = float('inf')
    else:
        it = load_dict.get('it', -1)
        epoch_idx = load_dict.get('epoch_idx', -1)
        fid_best = load_dict.get('fid_best', float('inf'))
        kid_best = load_dict.get('kid_best', float('inf'))
        logger.load_stats(stats_file)

    # Reinitialize model average if needed
    if (config['training']['take_model_average']
      and config['training']['model_average_reinit']):
        update_average(generator_test, generator_1, 0.)

    # Learning rate anneling
    # 3
    d_lr_1= d_optimizer_1.param_groups[0]['lr']
    g_lr_1 = g_optimizer_1.param_groups[0]['lr']
    g_scheduler_1 = build_lr_scheduler(g_optimizer_1, config, last_epoch=it)
    d_scheduler_1 = build_lr_scheduler(d_optimizer_1, config, last_epoch=it)
    # ensure lr is not decreased again
    d_optimizer_1.param_groups[0]['lr'] = d_lr_1
    g_optimizer_1.param_groups[0]['lr'] = g_lr_1
    
    d_lr_2= d_optimizer_2.param_groups[0]['lr']
    g_lr_2 = g_optimizer_2.param_groups[0]['lr']
    g_scheduler_2 = build_lr_scheduler(g_optimizer_2, config, last_epoch=it)
    d_scheduler_2 = build_lr_scheduler(d_optimizer_2, config, last_epoch=it)
    # ensure lr is not decreased again
    d_optimizer_2.param_groups[0]['lr'] = d_lr_2
    g_optimizer_2.param_groups[0]['lr'] = g_lr_2
    
    d_lr_3= d_optimizer_3.param_groups[0]['lr']
    g_lr_3 = g_optimizer_3.param_groups[0]['lr']
    g_scheduler_3 = build_lr_scheduler(g_optimizer_3, config, last_epoch=it)
    d_scheduler_3 = build_lr_scheduler(d_optimizer_3, config, last_epoch=it)
    # ensure lr is not decreased again
    d_optimizer_3.param_groups[0]['lr'] = d_lr_3
    g_optimizer_3.param_groups[0]['lr'] = g_lr_3

    # Trainer
    trainer_1 = Trainer(
        generator_1, discriminator_1, g_optimizer_1, d_optimizer_1,
        use_amp=config['training']['use_amp'],
        gan_type=config['training']['gan_type'],
        reg_type=config['training']['reg_type'],
        reg_param=config['training']['reg_param'],
        aug_policy=config['training']['aug_policy']
    )
    trainer_2 = Trainer(
        generator_2, discriminator_2, g_optimizer_2, d_optimizer_2,
        use_amp=config['training']['use_amp'],
        gan_type=config['training']['gan_type'],
        reg_type=config['training']['reg_type'],
        reg_param=config['training']['reg_param'],
        aug_policy=config['training']['aug_policy']
    )
    trainer_3 = Trainer(
        generator_3, discriminator_3, g_optimizer_3, d_optimizer_3,
        use_amp=config['training']['use_amp'],
        gan_type=config['training']['gan_type'],
        reg_type=config['training']['reg_type'],
        reg_param=config['training']['reg_param'],
        aug_policy=config['training']['aug_policy']
    )
    print('it {}: start with LR:\n\td_lr: {}\tg_lr: {}'.format(it, d_optimizer_1.param_groups[0]['lr'],g_optimizer_1.param_groups[0]['lr']))
    print('it {}: start with LR:\n\td_lr: {}\tg_lr: {}'.format(it, d_optimizer_2.param_groups[0]['lr'],g_optimizer_2.param_groups[0]['lr']))
    print('it {}: start with LR:\n\td_lr: {}\tg_lr: {}'.format(it, d_optimizer_3.param_groups[0]['lr'],g_optimizer_3.param_groups[0]['lr']))
    # Training loop
    print('Start training...')
    while True:
        epoch_idx += 1
        print('Start epoch %d...' % epoch_idx)

        for x_real in train_loader:
            t_it = time.time()
            it += 1
            generator_1.ray_sampler.iterations = it# for scale annealing
            generator_2.ray_sampler.iterations = it
            generator_3.ray_sampler.iterations = it
            #3
            # Sample patches for real data
            rgbs = img_to_patch(x_real.to(device))          # N_samples x C

            # Discriminator updates
            z = zdist.sample((batch_size,))
            dloss_1, reg_1 = trainer_1.discriminator_trainstep(rgbs, y=y, z=z, data_aug=config['data']['augmentation'])
            dloss_2, reg_2 = trainer_2.discriminator_trainstep(rgbs, y=y, z=z, data_aug=config['data']['augmentation'])
            dloss_3, reg_3 = trainer_3.discriminator_trainstep(rgbs, y=y, z=z, data_aug=config['data']['augmentation'])
            dloss=dloss_1+dloss_2+dloss_3
            reg=reg_1+reg_2+reg_3
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            if config['nerf']['decrease_noise']:
                generator_1.decrease_nerf_noise(it)
                generator_2.decrease_nerf_noise(it)
                generator_3.decrease_nerf_noise(it)
            #3
            
            z = zdist.sample((batch_size,))
            gloss_1 = trainer_1.generator_trainstep(y=y, z=z)
            gloss_2 = trainer_2.generator_trainstep(y=y, z=z)
            gloss_3 = trainer_3.generator_trainstep(y=y, z=z)
            gloss = gloss_1 + gloss_2 + gloss_3
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator_1,
                               beta=config['training']['model_average_beta'])
                update_average(generator_test, generator_2,
                               beta=config['training']['model_average_beta'])
                update_average(generator_test, generator_3,
                               beta=config['training']['model_average_beta'])
            #3
            
            # Update learning rate
            #3
            g_scheduler_1.step()
            d_scheduler_1.step()
            g_scheduler_2.step()
            d_scheduler_2.step()
            g_scheduler_3.step()
            d_scheduler_3.step()
            #3
            d_lr_1 = d_optimizer_1.param_groups[0]['lr']
            g_lr_1 = g_optimizer_1.param_groups[0]['lr']
            d_lr_2 = d_optimizer_2.param_groups[0]['lr']
            g_lr_2 = g_optimizer_2.param_groups[0]['lr']
            d_lr_3 = d_optimizer_3.param_groups[0]['lr']
            g_lr_3 = g_optimizer_3.param_groups[0]['lr']
            #
            logger.add('learning_rates', 'discriminator', d_lr_1, it=it)
            logger.add('learning_rates', 'generator', g_lr_1, it=it)

            dt = time.time() - t_it
            # Print stats
            if ((it + 1) % config['training']['print_every']) == 0:
                g_loss_last = logger.get_last('losses', 'generator')
                d_loss_last = logger.get_last('losses', 'discriminator')
                d_reg_last = logger.get_last('losses', 'regularizer')
                print('[%s epoch %0d, it %4d, t %0.3f] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                      % (config['expname'], epoch_idx, it + 1, dt, g_loss_last, d_loss_last, d_reg_last))

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                rgb, depth, acc = evaluator.create_samples(ztest.to(device), poses=ptest)
                logger.add_imgs(rgb, 'rgb', it)
                logger.add_imgs(depth, 'depth', it)
                logger.add_imgs(acc, 'acc', it)

            # (v) Compute fid if necessary
            if fid_every > 0 and ((it + 1) % fid_every) == 0:
                fid, kid = evaluator.compute_fid_kid()
                logger.add('validation', 'fid', fid, it=it)
                logger.add('validation', 'kid', kid, it=it)
                torch.cuda.empty_cache()
                gc.collect()
                # save best model
                if save_best=='fid' and fid < fid_best:
                    fid_best = fid
                    print('Saving best model...')
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best,
                                       kid_best=kid_best)
                    logger.save_stats('stats_best.p')
                    torch.cuda.empty_cache()
                    gc.collect()
                elif save_best=='kid' and kid < kid_best:
                    kid_best = kid
                    print('Saving best model...')
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best,
                                       kid_best=kid_best)
                    logger.save_stats('stats_best.p')
                    torch.cuda.empty_cache()
                    gc.collect()

            # (vi) Create video if necessary
            if ((it+1) % config['training']['video_every']) == 0:
                N_samples = 4
                zvid = zdist.sample((N_samples,))

                basename = os.path.join(out_dir, '{}_{:06d}_'.format(os.path.basename(config['expname']), it))
                evaluator.make_video(basename, zvid, render_poses, as_gif=False)

            # (i) Backup if necessary
            if ((it + 1) % backup_every) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx, fid_best=fid_best,
                                   kid_best=kid_best)
                logger.save_stats('stats_%08d.p' % it)

            # (vi) Save checkpoint if necessary
            if time.time() - t0 > save_every:
                print('Saving checkpoint...')
                checkpoint_io.save(model_file, it=it, epoch_idx=epoch_idx, fid_best=fid_best,
                                   kid_best=kid_best)
                logger.save_stats('stats.p')
                t0 = time.time()

                if (restart_every > 0 and t0 - tstart > restart_every):
                    exit(3)
