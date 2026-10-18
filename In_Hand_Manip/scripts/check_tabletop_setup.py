"""Replay all tabletop starts without resets; verify contacts, forces and task failures."""
import isaacgym
import argparse
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_manip import ROOT
from hora.utils.misc import set_seed
from hora.tasks.allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.tabletop_contacts import object_contact_summary
from scripts.check_grasp_cache import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=214)
    args = parser.parse_args()
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
        cfg = compose(config_name='manip_tabletop_box45', overrides=[
            'stage=teacher','num_envs=16','pipeline=cpu','rl_device=cpu','graphics_device_id=-1'])
    bank = torch.load(ROOT/cfg.task.env.grasp.cache,map_location='cpu')
    name = bank['objects'][0]['name']
    states,targets = bank['states'][name],bank['targets'][name]
    task = OmegaConf.to_container(cfg.task,resolve=True)
    task['stage'] = 'teacher'
    task['env']['numEnvs'] = len(states)
    set_seed(args.seed)
    env = AllegroDigitTabletop(task,'cuda:0',-1,True)
    env.stage = 'grasps'  # preserve failures rather than silently auto-reset
    try:
        install(env,states,targets)
        n = env.num_envs
        failed = torch.zeros(n,dtype=torch.bool)
        support = torch.zeros(n)
        two_gels = torch.zeros(n)
        non_gel = torch.zeros(n)
        maxforce = torch.zeros(n)
        maxoffset = torch.zeros(n)
        forbidden_count = 0
        for _ in range(args.steps):
            env.step(torch.zeros_like(env.actions))
            contacts = object_contact_summary(env)
            forces = env.contact_forces[:,env.gel_ids].norm(dim=-1)
            effective = contacts['gels'] & (forces > cfg.task.env.digit.contact_threshold)
            support += contacts['table'].float()
            two_gels += (effective.sum(-1)>=2).float()
            non_gel += contacts['other_hand'].float()
            maxforce = torch.maximum(maxforce,forces.max(-1).values)
            maxoffset = torch.maximum(maxoffset,(env.object_state[:,:2]-states[:,16:18]).norm(dim=-1))
            failed |= env.terminal['failed']
            forbidden_count += len(contacts['forbidden'])
        report = dict(count=n,seed=args.seed,domain_params=env.domain_params.tolist(),
                      seconds=args.steps*env.control_dt,failed=failed.tolist(),
                      table_support_fraction=(support/args.steps).tolist(),
                      two_object_gels_fraction=(two_gels/args.steps).tolist(),
                      non_gel_object_fraction=(non_gel/args.steps).tolist(),
                      max_gel_force_N=maxforce.tolist(),max_xy_drift_m=maxoffset.tolist(),
                      forbidden_contact_count=forbidden_count,
                      hand_shape_count=env.gym.get_asset_rigid_shape_count(env.hand_asset),
                      table_filter=env.collision_spec['table_filter'],
                      collision_sha256=env.collision_hash)
        report['passed'] = bool(not failed.any() and forbidden_count==0 and
                                (support/args.steps>=.9).all() and
                                (two_gels/args.steps>=.9).all() and
                                (non_gel/args.steps<=.1).all() and (maxforce<=2).all())
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not report['passed']:
            raise RuntimeError('Tabletop cached starts did not pass replay checks')
    finally:
        env.close()


if __name__ == '__main__':
    main()
