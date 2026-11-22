import asyncio
import json
from pathlib import Path
import random
import sys

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
NAME = sys.argv[2] if len(sys.argv) > 2 else f"random-{random.randint(100, 999)}"

# A deck is required to join: 8 different playable cards, at most one champion.
REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_PATH = REPO_ROOT / "gym" / "data" / "cards.json"
DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]
with CARDS_PATH.open() as file:
    card_definitions = json.load(file)["cards"]

ENTITY_TYPES = sorted(
    name
    for name, definition in card_definitions.items()
    if definition.get("type") in ("troop", "building")
)
# entitiy types represents the name all cards of type troop or building, plus unknown to deal with edge cases
ENTITY_TYPES.append("unknown")
# type index maps each name to its index in entity types, creating easy access
TYPE_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

POSITIONS = [
        (3, 14.5),
        (15, 14.5),
        (3, 8),
        (15, 8),
        (3, 5),
        (15, 5),
    ]

NUM_ACTIONS = 1 + 4 * len(POSITIONS) 

def policy(obs):
    """
    {
        "tick": 1240, 
        "phase": "single",
        "elixir": 6.4, 
        "crowns": [0, 1],
        "hand": ["valkyrie", "musketeer", "giant", "minipekka"],
        "hand_cost": [4, 4, 5, 4], 
        "next_card": "knight",
        "legal": [true, true, true, true],
        "last_action": "ok",
        "entities": [[0, "king_tower", 9.0, 3.0, 4008, 4008, false, true, false, 1, 2.0, 2.0], ...],
        "done": false, "outcome": 0
    }
    """
    model_input = encode_observation(obs)
    mask = action_mask(obs)
    allowed = [i for i, valid in enumerate(mask) if valid]
    action_id = random.choice(allowed)
    action = decode_action(action_id)
    return action

def encode_observation(obs):
    model_input = []
    model_input.append(obs["elixir"] / 10)
    # Stores list of binary numbers [0 0 1] representing which phase game is in
    model_input.extend([
        float(obs["phase"] == phase) for phase in ("single", "double", "overtime")
    ])
    # appends 32 entries - each 8 binary values [0 0 0 0 0 0 1 0] representing whether cards from deck are in our hand
    for card in obs["hand"]:
        model_input.extend([
            float(card == name) for name in DECK
        ])
    # store tower health with 6 entries - default 0 in case one is destroyed
    tower_health = [0.0] * 6
    # use obs["entities"] to get percent hp remaining for each tower, stored by index
    for entry in obs["entities"]:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entry

        if not is_tower:
            continue
        if card == "king_tower":
            tower_index = 2
        else:
            tower_index = 0 if x < 9 else 1

        index = owner * 3 + tower_index
        tower_health[index] = hp / max_hp if max_hp > 0 else 0.0
    model_input.extend(tower_health)
    # create a list representing how many of each troop type there are in certain locations; eight locations, - default 0
    counts = [
        [[0.0] * len(ENTITY_TYPES) for _ in range(8)]
        for _ in range(2)
    ]
    for entry in obs["entities"]:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entry

        if is_tower:
            continue
        # lane represents left/right with 0/1
        # band represents which fourth of the map to place with int 0-3; min max logic ensures it never goes beyond this range (32 / 8 divides into 4 even sections)
        lane = 0 if x < 9 else 1
        band = max(0, min(3, int(y / 8)))
        # multiplying by two and occasionally adding one guarantees all unique numbers from 0-7
        region = band * 2 + lane
        # use type index to get the index of the card in entity types
        troop_type = TYPE_INDEX.get(card, TYPE_INDEX["unknown"])
        # update counter in counts, which stores how many of whos troop type there are in which area 
        counts[owner][region][troop_type] += 1
    # add every count value to the list
    model_input.extend([
        count 
        for owner_counts in counts 
        for region_counts in owner_counts 
        for count in region_counts
    ])

    return model_input
    # Your king tower is at (9, 3) and your princess towers at (3.5, 6.5) and (14.5, 6.5). The opponent's are at (9, 29), (3.5, 25.5) and (14.5, 25.5).

def decode_action(action_id):
    # takes action id and converts to move - slot, x, y (or none if passing)
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"Invalid action: {action_id}")
    
    if action_id == 0:
        return None

    slot, position_index = divmod(action_id - 1, len(POSITIONS))
    x, y = POSITIONS[position_index]

    return (slot, x, y)

def action_mask(obs):
    # This creates a mask, with the first True representing the move pass
    mask = [True]
    # obs legal returns legal cards; adds each cards true/false value multiple times to cover all moves using that slot
    for affordable in obs["legal"]:
        mask.extend([affordable] * len(POSITIONS))

    return mask


async def play():
    """One connection: join, then answer until it closes. Returns False if the server refused us."""
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"t": "join", "name": NAME, "deck": DECK}))
        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("t")
            if t == "obs":
                # answer every tick, and say which tick the answer is for
                obs = msg["obs"]
                await ws.send(json.dumps({"t": "act", "tick": obs["tick"], "action": policy(obs)}))
            elif t == "joined":
                print(f"joined as {msg['name']} (rating {msg['rating']})")
            elif t == "start":
                # a tournament series can deal a deck for one game; the hand already shows it
                dealt = msg.get("deck")
                print(f"match vs {msg['opponent']}" + (f" with a dealt deck: {', '.join(dealt)}" if dealt else ""))
            elif t == "result":
                print(f"  {msg['result'].upper()} {msg['crowns']}  rating {msg['rating']} "
                      f"({msg['delta']:+})  replay: /replay?id={msg['match']}")
            elif t == "aborted":
                print(f"  match void: {msg['reason']}")
            elif t == "error":
                print("error:", msg.get("msg"))
                return False
    return True


async def main():
    # Stay in the pool for the whole tournament. If the connection drops, join
    # again under the same name: that is what keeps our place in the bracket.
    while True:
        try:
            if not await play():
                break
        except (OSError, websockets.WebSocketException) as error:
            print("connection lost:", error)
        print("joining again in 5 s")
        await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass