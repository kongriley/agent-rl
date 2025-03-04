from utils.self_bleu import SelfBleuReward



def test_self_bleu():
    references = ["The quick brown fox jumps over the lazy dog", "The fast brown fox jumps over the lazy dog"]
    reward_module = SelfBleuReward()
    for r in references:
        reward_module.append_reference(r)
    for r in references:
        score = reward_module(r)
        print(score)

test_self_bleu()