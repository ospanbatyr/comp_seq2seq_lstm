class BeamSearchNode(object):
    """
    The node class for performing beam search
    """
    def __init__(self, hidden_state, previous_node, word_id, log_prob, length):
        self.h = hidden_state
        self.prev_node = previous_node
        self.word_id = word_id
        self.logp = log_prob
        self.leng = length

    def eval(self, repeat_penalty=0, token_reward=0, score_table=None, alpha=1.0):
        reward = 0
        # add here a function for shaping a reward 

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    




        