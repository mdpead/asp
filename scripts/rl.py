from src import data, model, tokenizer, utils
import logging


# Reinforcement learning on synth tasks with verifiable rewards: sample rollouts from a
# prompt, score them by executing the generated function, update on the advantage. Runs the
# data path today; the rollout loop is not wired up yet — see the TODOs at the bottom.
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "rl")

    token = tokenizer.load_tokenizer(utils.get_run_path(config))

    ds_raw = data.get_dataset_rl(config)

    ds = data.prepare_rl(ds_raw, token, config)

    logging.info(f"rl prompts: {  {split: len(rows) for split, rows in ds.items()} }")

    model.build_transformer(config)

    # TODO rollout sampling. generation.py decodes greedily (argmax at generation.py:88), so
    # every rollout in a group would be identical; RL needs temperature and top-p first.

    # TODO the loop itself. This is not train.train(): there are no fixed targets, so it is
    # generate a group of K rollouts per prompt -> score each with
    # synth.check_output_answer / check_input_answer -> group-relative advantage -> policy
    # gradient step. GRPO fits, since the rewards are binary and verifiable and no value
    # network is needed.

    # TODO initialise from the SFT checkpoint under models/<name>/sft/checkpoints/, model
    # weights only, same gap as in scripts/sft.py.

    logging.warning("RL data path ran; the rollout loop is not wired up yet (see TODOs)")


if __name__ == "__main__":
    main()
