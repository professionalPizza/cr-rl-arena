import asyncio
import json
import random
import sys

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
NAME = sys.argv[2] if len(sys.argv) > 2 else f"random-{random.randint(100, 999)}"


# A deck is required to join: 8 different playable cards, at most one champion.
DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"]


def policy(obs):
    """
    {
        "tick": 1240, "phase": "single",
        "elixir": 6.4, "crowns": [0, 1],
        "hand": ["valkyrie", "musketeer", "giant", "minipekka"],
        "hand_cost": [4, 4, 5, 4], "next_card": "knight",
        "legal": [true, true, true, true],
        "last_action": "ok",
        "entities": [[0, "king_tower", 9.0, 3.0, 4008, 4008, false, true, false, 1, 2.0, 2.0], ...],
        "done": false, "outcome": 0
    }
    """
    counter_preferences = {
        "giant": ["minipekka", "minions", "archers"],
        "knight": ["musketeer", "minions", "knight"],
        "archers": ["wizard", "valkyrie"],
        "valkyrie": ["knight", "minions", "minipekka"],
        "skeleton_army": ["valkyrie", "wizard"],
        "minions": ["wizard", "archers"],
        "balloon": ["musketeer", "minions", "archers"],
        "minipekka": ["minions", "knight"],
        "pekka": ["minions"],
        "musketeer": ["knight", "valkyrie"],
        "wizard": ["knight", "valkyrie"],
        "babydragon": ["musketeer"],
    }
    move_options = [[3, 15], [15, 15]]                                      # by default, play offense
    chosen_move = random.choice(move_options)
    elixir_threshold = 5.5
    legal = obs.get("legal", [])
    playable = [i for i, ok in enumerate(legal) if ok]                      # extracts legal slot indexes
    hand = obs["hand"]
    enemies = [i for i in obs["entities"] if i[0] == 1]
    allies = [i for i in obs["entities"] if i[0] == 0]
    slot = -1
    for ally in allies:
        ally_y = ally[3]
        if ally_y > 15:
            elixir_threshold = 0
            if ally[2] > 9.5:
                chosen_move = move_options[1]
            else:
                chosen_move = move_options[0]
    for enemy in enemies:
        enemy_y = enemy[3]
        if enemy_y < 15:
            elixir_threshold = 0
            enemy_type = enemy[1]
            counter_options = counter_preferences.get(enemy_type, [])
            if enemy[2] > 9.5:
                chosen_move = [15, enemy_y - 2]
            else:
                chosen_move = [3, enemy_y - 2]
            for option in counter_options:
                if slot != -1:
                    continue
                for index in playable:
                    if hand[index] == option:
                        slot = index
                        break
    if obs["elixir"] < elixir_threshold:
        return None
    if not playable or random.random() < 0.3:                               # if playable is empty, or random < .6,
        return None                                                         # Pass; otherwise, 
    if slot == -1:
        slot = random.choice(playable)                                      # Randomly select from playable moves
    x = chosen_move[0]
    y = chosen_move[1]
    return [slot, x, y]     # Returns randomly selected slot, predetermined x, y


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
