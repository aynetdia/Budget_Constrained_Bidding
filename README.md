# Ad Tech Budget Constrained Real-Time Bidding (RTB)
This repo contains the implementation of the DRLB framework introduced by Wu et al. (2018).

### Configuration parameters

Set the `src/gym_auction_emulator/gym_auction_emulator/envs/config.cfg` accordingly to training or testing mode. Currently it is set to testing mode to allow for tunning the evaluation of the DRLB by our peers.

The `src/rtb_agent/config.cfg` also has to be adjusted depending on which type of the experiment you want to run. The experiment are pretty self-explanatory, when coupled with the section 5.4 of the seminar paper. 

## How to Use:

The training of the DRLB can be done from the console by executing either the .sh script as specified below or running the src/rl_bid_agent.py script as a python script, after adjusting the config files.

Use the following command to train the RL Bidding Agent.
<br />```bash train_rl_bid.sh```

The evaluation can be done by running the evaluation script in the `src/rtb_agent/` folder. The names of the evaluation scripts are tied to the names of the models used in the seminar paper

## Access to data

The iPinYou dataset can be downloaded and preprocessed using the following github repo: https://github.com/wnzhang/make-ipinyou-data

In order to properly run all the experiments the data frames with predicted CTRs can be downloaded from: https://drive.google.com/drive/folders/1dsPV_vDNNdxhy9ZUsk75RX7HowzNsRaC?usp=sharing instead of using the trained logistic regression from `src/rtb_agent/logreg.sav` to get predictions on your own. Then the `data/supplement.ipynb` script can be used to insert the pCTR column into the preprocessed dataset.

## Repository Citations

[Wu, Di, et al. "Budget constrained bidding by model-free reinforcement learning in display advertising." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. ACM, 2018.](https://arxiv.org/pdf/1802.08365)
