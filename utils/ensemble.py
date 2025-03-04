def ensemble_vote(predictions):
    """
    Given a list of predictions [(class, confidence), ...], return the voted class and average confidence.
    For simplicity, we choose the class that appears most frequently.
    """
    vote_count = {}
    total_conf = {}
    for cls, conf in predictions:
        vote_count[cls] = vote_count.get(cls, 0) + 1
        total_conf[cls] = total_conf.get(cls, 0) + conf
    
    # Find class with maximum votes
    final_class = max(vote_count, key=vote_count.get)
    final_conf = total_conf[final_class] / vote_count[final_class]
    return final_class, final_conf

if __name__ == "__main__":
    preds = [("car", 0.8), ("car", 0.7), ("bicycle", 0.9)]
    print(ensemble_vote(preds))
