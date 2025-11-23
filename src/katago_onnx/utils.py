import pathlib

import numpy as np
import pandas as pd
import torch

from katago.game.data import load_sgf_moves_exn
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train.load_model import load_model as load_model_katago


def load_model(model_path: str | pathlib.Path, device: str = "cpu"):
    model, swa_model, _ = load_model_katago(
        str(model_path),
        use_swa=False,
        device=device,
        pos_len=19,
        verbose=True,
    )

    model = model.eval()

    return model


def load_sgf(sgf_path: str | pathlib.Path, target_move: int | None = None):
    # Load SGF
    metadata, setup, moves, rules = load_sgf_moves_exn(str(sgf_path))
    print(f"Loaded SGF with {len(moves)} moves.")
    print(f"Game size: {metadata.size}")

    # Initialize GameState
    board_size = metadata.size
    rules = GameState.RULES_JAPANESE
    gamestate = GameState(board_size, rules)

    if target_move is None:
        target_move = len(moves)

    # Play moves
    for i, (pla, loc) in enumerate(moves):
        if i >= target_move:
            break
        gamestate.play(pla, loc)

    print(f"Replayed to move {len(gamestate.moves)}")

    return gamestate


def featurize(gamestate: GameState, model, device: str | None = None):
    if type(gamestate.board_size) is int:
        pos_len = gamestate.board_size
    else:
        pos_len = gamestate.board_size[0]  # type: ignore

    # Prepare features
    features = Features(model.config, pos_len=pos_len)
    bin_input, global_input = gamestate.get_input_features(features)

    if device is None:
        device = model.device

    # Convert to torch
    bin_input = torch.tensor(bin_input).to(device)
    global_input = torch.tensor(global_input).to(device)

    return (
        features,
        bin_input,
        global_input,
    )


def run_inference(
    model,
    bin_input: torch.Tensor,
    global_input: torch.Tensor,
):
    # Run model
    with torch.no_grad():
        outputs = model(bin_input, global_input)

    # Unpack outputs
    if model.has_intermediate_head:
        main_outputs, intermediate_outputs = outputs
    else:
        main_outputs = outputs[0]

    (
        policy_logits,
        value_logits,
        miscvalue_logits,
        moremiscvalue_logits,
        ownership_logits,
        scoring_logits,
        futurepos_logits,
        seki_logits,
        scorebelief_logits,
    ) = main_outputs

    # Process Policy
    # policy_logits shape: [batch, num_policy_outputs, num_moves]
    # We want the first policy output (index 0) and softmax over moves (dim 1)
    policy_probs = torch.nn.functional.softmax(policy_logits[:, 0, :], dim=1).cpu().numpy()[0]

    # Process Value (Winrate)
    # value_logits shape is usually [batch, 3] -> [win, loss, no_result] or similar depending on config
    # But typically for KataGo:
    # value_output is [batch, 3] corresponding to [win, loss, no_result] or similar.

    # Actually, let's check model_pytorch.py or just print shape.
    value_probs = torch.nn.functional.softmax(value_logits, dim=1).cpu().numpy()[0]

    winrate = value_probs[0]  # Assuming index 0 is win for current player? Or black? Or white?

    # Process Score Lead
    # miscvalue_logits shape: [batch, 10]
    # Index 2 is lead
    score_lead = miscvalue_logits[:, 2].item() * model.lead_multiplier

    return policy_probs, value_probs, winrate, score_lead


def get_top_moves(
    policy_probs: np.ndarray,
    gamestate: GameState,
    features: Features,
    top_k: int = 10,
    model=None,
):
    top_moves_idx = np.argsort(policy_probs)[::-1][:top_k]

    top_moves = []
    for idx in top_moves_idx:
        loc = features.tensor_pos_to_loc(idx, gamestate.board)

        x, y = None, None
        move_str = ""

        if loc is None:
            move_str = "PASS"
        elif loc == gamestate.board.loc(-10, -10):  # Illegal/Offboard
            move_str = "Illegal"
        else:
            x = gamestate.board.loc_x(loc)
            y = gamestate.board.loc_y(loc)
            # Convert to SGF coordinates (e.g. A1, D4) or similar
            # KataGo board uses 0-indexed coordinates.
            # Let's just print (x,y) for now or convert to GTP-like string
            col_str = "ABCDEFGHJKLMNOPQRST"[x]
            row_str = str(gamestate.board.y_size - y)
            move_str = f"{col_str}{row_str}"

        move_data = {
            "loc": loc,
            "x": x,
            "y": y,
            "prob": policy_probs[idx],
            "move_str": move_str,
        }

        # If model is provided, compute winrate and score lead for this move
        if model is not None and loc != gamestate.board.loc(-10, -10):
            # Clone gamestate
            # Note: GameState.copy() might not be deep enough or exist, let's check board.py
            # Assuming we can copy or re-create.
            # Actually GameState has a copy method? Let's check gamestate.py
            # For now, let's assume we can just play on a copy.

            # Create a new gamestate from scratch is safer if copy is not robust
            # But we need the history.
            # Let's try to use the copy method if it exists.
            gs_copy = gamestate.copy()

            try:
                gs_copy.play(gs_copy.board.pla, loc)

                # Featurize
                _, bin_input, global_input = featurize(gs_copy, model)

                # Run inference
                _, _, winrate, score_lead = run_inference(model, bin_input, global_input)

                # The winrate/score is from the perspective of the NEXT player (who just played? No, who is TO PLAY)
                # After playing, it's the opponent's turn.
                # So the value returned is for the opponent.
                # We want it for the current player.
                move_data["score_lead"] = -score_lead

            except Exception as e:
                print(f"Error computing value for move {move_str}: {e}")
                move_data["score_lead"] = np.nan

        top_moves.append(move_data)

    return pd.DataFrame(top_moves)
