# --- 1. CONFIGURATION ---
GESTURES = ["rock", "paper", "scissors"]
GESTURE_ID_MAP = {
    0: 'rock', 
    1: 'paper', 
    2: 'scissors'
}
YOLO_MODEL_PATH = "best.pt"  # Your model file

# --- 2. HAND DATA STRUCTURE ---
class Hand:
    """Represents a single detected hand gesture with its location."""
    def __init__(self, gesture_id, x_center, y_center, width, height):
        self.gesture_id = gesture_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.player = None  
        self.side = None    

# --- 3. GAME LOGIC FUNCTIONS ---
def beats(a, b):
    # RPS win logic
    return (a == "rock" and b == "scissors") or \
           (a == "scissors" and b == "paper") or \
           (a == "paper" and b == "rock")

def remove_gesture(my_hand, opp_hand):
    # Your complex RPS-Minus-One removal logic (Rules 2, 3, 4)
    # ... (Keep the original implementation here) ...
    my_set = set(my_hand)
    opp_set = set(opp_hand)

    common = list(my_set & opp_set)
    my_unique = list(my_set - opp_set)
    opp_unique = list(opp_set - my_set)

    if len(common) == 2:
        a, b = list(my_set)
        if beats(a, b):
            return b
        else:
            return a

    if len(common) == 1:
        
        # --- FIX: Ensure unique lists have exactly one element ---
        if len(my_unique) != 1 or len(opp_unique) != 1:
            print("WARNING: Player hand input violates game rules (not exactly 2 distinct gestures). Falling back to arbitrary choice.")
            return random.choice(my_hand)
        # --------------------------------------------------------

        common_gesture = common[0]
        my_nc = my_unique[0]
        opp_nc = opp_unique[0]

        # Rule 2 — your non-common wins → remove your non-common
        if beats(my_nc, opp_nc):
            return my_nc

        # Rule 4 — your non-common loses → remove common gesture
        if beats(opp_nc, my_nc):
            return common_gesture

    # Handle the "No common gesture" case (len(common) == 0)
    if len(common) == 0:
        # If your set is {R, P} and opp set is {S, C} (impossible in RPS), 
        # but if your set is {R, P} and opp set is {S, C} this block will be hit.
        # Your original code falls back here, so we add a specific rule for this.
        # A common default is to remove the hand that loses to the opponent's best hand.
        return random.choice(my_hand) # Arbitrary choice for non-standard case

    return 'rock' # Fallback (arbitrary choice)
# ----------------------------------------------------

from ultralytics import YOLO
import argparse
import random

# --- 4. DATA EXTRACTION AND PARSING (New Function) ---

def detect_hands(image_path, model_path, gesture_map):
    """
    Loads the YOLO model, runs inference on the image, and converts results 
    into a list of Hand objects using normalized coordinates.
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Ensure 'ultralytics' is installed and the file exists.")
        print(f"Details: {e}")
        return []

    # Run inference
    results = model(image_path, verbose=False) 
    
    hands = []
    
    # Assuming batch size of 1, process the first result object
    if results and results[0].boxes:
        r = results[0]
        img_width, img_height = r.orig_shape # Get the original image size

        for box in r.boxes:
            # box.xywhn returns: [x_center, y_center, width, height] (normalized 0-1)
            x_c_norm, y_c_norm, w_norm, h_norm = box.xywhn[0].tolist() 
            class_id = int(box.cls[0].item())

            # Only process if the class ID is in our defined map
            if class_id in gesture_map:
                hands.append(Hand(
                    gesture_id=class_id,
                    x_center=x_c_norm,
                    y_center=y_c_norm,
                    width=w_norm,
                    height=h_norm
                ))
    
    return hands

# --- 5. RPS Solver Class (Same as before) ---
class RPSCustomSolver:
    
    def __init__(self, hands):
        self.hands = hands
        self.gesture_names = GESTURE_ID_MAP 

    def _assign_hands_to_players(self):
        # ... (Same logic as before to assign 'MINE', 'OPPONENT', 'LEFT', 'RIGHT')
        opponent_hands = []
        my_hands = []
        
        for hand in self.hands:
            # Y-center < 0.5 (top half) is OPPONENT, Y-center > 0.5 (bottom half) is MINE
            if hand.y_center < 0.5: 
                hand.player = 'OPPONENT'
                opponent_hands.append(hand)
            else:
                hand.player = 'MINE'
                my_hands.append(hand)

        my_hands.sort(key=lambda h: h.x_center)
        if len(my_hands) >= 2:
            my_hands[0].side = 'LEFT'
            my_hands[1].side = 'RIGHT'
        
        opponent_hands.sort(key=lambda h: h.x_center)
        
        self.opponent_hands = opponent_hands
        self.my_hands = my_hands


    def solve(self):
        self._assign_hands_to_players()
        
        if len(self.opponent_hands) < 2 or len(self.my_hands) < 2:
            return f"⚠️ Error: Model detected {len(self.opponent_hands)} opponent hand(s) and {len(self.my_hands)} of your hand(s). Need two of each for the game."

        # Convert detected Hands into lists of gesture strings
        my_gesture_names = [self.gesture_names[h.gesture_id] for h in self.my_hands]
        opp_gesture_names = [self.gesture_names[h.gesture_id] for h in self.opponent_hands]
        
        # 1. Determine which hand to remove using your custom logic
        gesture_to_remove = remove_gesture(my_gesture_names, opp_gesture_names)

        # 2. Find which physical hand (Left/Right) corresponds to the gesture to remove
        hand_to_remove_side = None
        
        # We must handle the case where the gestures are identical
        if self.gesture_names[self.my_hands[0].gesture_id] == gesture_to_remove and self.gesture_names[self.my_hands[1].gesture_id] == gesture_to_remove:
            # Both hands are the gesture to remove (e.g., if the logic returns 'rock' but you only have 'paper', 'scissors')
            # This is technically an error state based on your rules, but we must choose one.
             hand_to_remove_side = 'LEFT' # Arbitrarily remove left
        elif self.gesture_names[self.my_hands[0].gesture_id] == gesture_to_remove:
            hand_to_remove_side = 'LEFT'
        elif self.gesture_names[self.my_hands[1].gesture_id] == gesture_to_remove:
            hand_to_remove_side = 'RIGHT'
        

        if hand_to_remove_side is None:
             return f"⚠️ Logic Error: Could not find the physical hand matching the required removal gesture: **{gesture_to_remove}**"

        # 3. Formulate the final instruction
        result_message = (
            f"\n--- Game Analysis ---\n"
            f"Your hands: **{my_gesture_names[0].capitalize()}** (Left), **{my_gesture_names[1].capitalize()}** (Right)\n"
            f"Opponent's hands: **{opp_gesture_names[0].capitalize()}**, **{opp_gesture_names[1].capitalize()}**\n"
            f"Custom Logic Determined Remove: **{gesture_to_remove.capitalize()}**\n"
            f"✅ **ACTION: REMOVE your {hand_to_remove_side} hand.**"
        )
        
        return result_message

# --- 6. MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Rock-Paper-Scissors-Minus-One Solver.")
    parser.add_argument("image_path", type=str, help="Path to the input image file (e.g., my_game_image.jpg).")
    args = parser.parse_args()

    # 1. Run detection on the uploaded image
    print(f"Running detection on {args.image_path} using model {YOLO_MODEL_PATH}...")
    hands = detect_hands(args.image_path, YOLO_MODEL_PATH, GESTURE_ID_MAP)

    if not hands:
        print("❌ No hands were detected in the image, or the model failed to load.")
        return

    # 2. Solve the game
    solver = RPSCustomSolver(hands)
    instruction = solver.solve()
    
    print(instruction)

if __name__ == "__main__":
    main()