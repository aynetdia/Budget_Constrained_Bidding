# Ad Tech Budget Constrained Real-Time Bidding (RTB)
This repo contains the implementation of the DRLB framework introduced by Wu et al. (2018).

## How to Use:

It is set up in such a way that training and (most importantly) testing works from the notebook. 

However, the training of the DRLB can also be done from the console by executing either the sh script as specified below or running the src/rl_bid_agent.py script as a python script. In both latter cases the main method in the script should be uncommented.

Use the following command to run the RL Bidding Agent.
<br />```bash scripts/run_rl_bid.sh```

### Configuration parameters

Set the `src/gym_auction_emulator/gym_auction_emulator/envs/config.cfg` accordingly to training or testing mode. Currently it is set to testing mode to allow for tunning the evaluation of the DRLB by our peers.

##§ Ad Exchange Open AI Gym environment

Open AI Gym environment mimics the Ad Exchange by taking the bid requests from the iPinYou or any other Bidding dataset. The agents can interact with it using the standard Gym API interface.

## Repository Citations

[Wu, Di, et al. "Budget constrained bidding by model-free reinforcement learning in display advertising." Proceedings of the 27th ACM International Conference on Information and Knowledge Management. ACM, 2018.](https://arxiv.org/pdf/1802.08365)
