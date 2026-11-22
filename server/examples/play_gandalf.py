import argparse
import asyncio
import json
from pathlib import Path

import torch
import websockets

from gandalf import (
    DECK, CARD_NAMES, BOARD_WIDTH, BOARD_HEIGHT, MAX_ENTITIES,
    encode_values, action_masks, decode_action,
)
from model_brains import ActorCritic, select_action
from gandalf_coordinate import CoordinateActorCritic, PlacementTemperature


def load_policy(args):
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    expected_config = {
        "encoding_version": 2,
        "action_version": 2,
        "architecture": "entity_pool_slot_tile_v1",
        "card_names": CARD_NAMES,
        "deck": DECK,
        "max_entities": MAX_ENTITIES,
        "board_width": BOARD_WIDTH,
        "board_height": BOARD_HEIGHT,
        "gamma": 0.9999,
    }
    config = checkpoint["config"]
    coordinate = config.get("architecture") == "entity_pool_coordinate_tile_v1"
    if coordinate:
        expected_config.update(architecture="entity_pool_coordinate_tile_v1", coordinate_hidden_size=32)
    if "decision_protocol" in config:
        expected_config["decision_protocol"] = "resource_wait_mc_v1"
    decks = {"original": DECK}
    if "deck_pool" in config:
        decks = {"original": DECK, "wizard": ["wizard" if c == "musketeer" else c for c in DECK]}
        expected_config.pop("deck")
        expected_config.update(deck_pool=decks, deck_protocol="original_wizard_mix_v1")
    if config != expected_config:
        raise ValueError("Unsupported checkpoint configuration; use a compatible battle checkpoint.")
    if args.deck not in decks:
        raise ValueError(f"Checkpoint supports decks: {list(decks)}")
    model = (CoordinateActorCritic if coordinate else ActorCritic)(len(CARD_NAMES))
    model.load_state_dict(checkpoint["model"])
    policy = PlacementTemperature(model, args.temperature).eval()
    print(f"Loaded {args.checkpoint}; mixed updates={checkpoint.get('deck_mix_updates', 'n/a')}; "
          f"deck={args.deck}; placement temperature={args.temperature}")
    if "deck_mix_progress" in checkpoint:
        print("Saved decisions:", {k: v["decisions"] for k,v in checkpoint["deck_mix_progress"].items()})
    return policy, decks[args.deck]


async def play(args, model, deck, games_left):
    """One connection: join, then answer until it closes. Returns the games finished."""
    completed = 0
    replay_base = args.url.replace("ws", "http", 1).rstrip("/")

    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps({
            "t": "join",
            "name": args.name,
            "deck": deck,
        }))

        async for raw in ws:
            message = json.loads(raw)
            kind = message.get("t")

            if kind == "obs":
                obs = message["obs"]
                if obs.get("done"):
                    continue
                slot_mask, tile_masks = action_masks(obs)

                slot, tile, _, _ = select_action(
                    model,
                    encode_values(obs),
                    slot_mask,
                    tile_masks,
                )
                action = decode_action(slot, tile)

                await ws.send(json.dumps({
                    "t": "act",
                    "tick": obs["tick"],
                    "action": action,
                }))

                if slot != 0:
                    print(
                        f"Tick {obs['tick']}: "
                        f"{obs['hand'][slot - 1]} at "
                        f"({action[1]}, {action[2]})"
                    )

            elif kind == "joined":
                print(f"Connected as {message['name']}")

            elif kind == "start":
                # In a tournament series the server can deal a deck for one game
                # in place of the one we joined with. The hand in every
                # observation is already that deck's, so only say so.
                dealt = message.get("deck")
                print(
                    f"Playing against {message['opponent']}"
                    + (f" with a dealt deck: {', '.join(dealt)}" if dealt else "")
                )

            elif kind == "result":
                print(
                    f"{message['result']} — "
                    f"{replay_base}/replay?id={message['match']}"
                )
                completed += 1
                if games_left is not None and completed >= games_left:
                    break

            elif kind == "aborted":
                print("Match aborted:", message["reason"])

            elif kind == "error":
                raise RuntimeError(message["msg"])

    return completed


async def main(args):
    model, deck = load_policy(args)
    if args.check_only:
        return

    # Stay in the pool for the whole tournament: a pairing can be a best of 5
    # and there are several rounds, with long waits in between. If the
    # connection drops (the server restarts, the network blips), join again
    # under the same name, which is what keeps our place in the bracket.
    games_left = args.games
    while games_left is None or games_left > 0:
        try:
            completed = await play(args, model, deck, games_left)
            if games_left is not None:
                games_left -= completed
        except (OSError, websockets.WebSocketException, RuntimeError) as error:
            print(f"Connection lost: {error}")
        if games_left is None or games_left > 0:
            print("Joining again in 5 s")
            await asyncio.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path, default=Path(__file__).resolve().parents[2]/"checkpoints/gandalf/deck_mix_wizard_v1/latest.pt")
    parser.add_argument("--deck", choices=["original", "wizard"], default="original")
    parser.add_argument("--temperature", type=float, default=.25)
    parser.add_argument("--check-only", action="store_true", help="Validate loading without joining matches")
    parser.add_argument("--name", default="Gandalf-review")
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--games", type=int, default=None,
                        help="Stop after this many games (default: keep playing until stopped)")
    arguments = parser.parse_args()

    if arguments.games is not None and arguments.games < 1:
        parser.error("games must be positive")
    torch.set_num_threads(1)
    try:
        asyncio.run(main(arguments))
    except KeyboardInterrupt:
        pass
