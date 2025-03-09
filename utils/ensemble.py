import logging
from typing import List, Tuple, Dict

# Set up logging
logger = logging.getLogger(__name__)

def ensemble_vote(predictions: List[Tuple[str, float]]) -> Tuple[str, float]:
    """
    Given a list of predictions [(class, confidence), ...], return the voted class and average confidence.
    
    Args:
        predictions: List of tuples (class_name, confidence)
        
    Returns:
        Tuple of (final_class, final_confidence)
    """
    if not predictions:
        logger.warning("Empty predictions list provided to ensemble_vote")
        return "unknown", 0.0
        
    try:
        # Count votes and sum confidences by class
        vote_count: Dict[str, int] = {}
        total_conf: Dict[str, float] = {}
        
        for cls, conf in predictions:
            if cls not in vote_count:
                vote_count[cls] = 0
                total_conf[cls] = 0.0
            
            vote_count[cls] += 1
            total_conf[cls] += conf
        
        if not vote_count:
            logger.warning("No valid votes collected")
            return "unknown", 0.0
            
        # Find class with maximum votes
        final_class = max(vote_count, key=vote_count.get)
        final_conf = total_conf[final_class] / vote_count[final_class]
        
        # Log voting details
        vote_details = ", ".join([f"{cls}({count})" for cls, count in vote_count.items()])
        logger.debug(f"Ensemble vote: {vote_details} -> {final_class} ({final_conf:.2f})")
        
        return final_class, final_conf
        
    except Exception as e:
        logger.error(f"Error in ensemble voting: {e}")
        return "unknown", 0.0

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test Ensemble Voting")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Test cases
    test_cases = [
        [("car", 0.8), ("car", 0.7), ("bicycle", 0.9)],
        [("person", 0.4), ("dog", 0.3), ("person", 0.5), ("person", 0.6)],
        [("unknown", 0.1), ("car", 0.2), ("bicycle", 0.15)],
        []  # Empty case
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {test_case}")
        
        result = ensemble_vote(test_case)
        print(f"Result: {result[0]} with confidence {result[1]:.4f}")
