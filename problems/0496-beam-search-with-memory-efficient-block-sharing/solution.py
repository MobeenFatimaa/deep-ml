import math
import numpy as np


class Block:

    def __init__(self, block_id, tokens=None):
        self.block_id = block_id
        self.tokens = list(tokens) if tokens is not None else []
        self.ref_count = 1

    def is_full(self, max_size):
        return len(self.tokens) >= max_size


class Beam:

    def __init__(self, score, block_table, finished=False):
        self.score = score
        self.block_table = list(block_table)
        self.finished = finished

    def get_tokens(self):
        tokens = []
        for block in self.block_table:
            tokens.extend(block.tokens)
        return tokens


def beam_search_block_sharing(
    log_probs, beam_width, block_size, eos_token=-1
):
    """Perform beam search decoding with memory-efficient block sharing."""
    max_steps, vocab_size = log_probs.shape
    if max_steps == 0 or beam_width == 0:
        return {
            "sequences": [],
            "scores": [],
            "total_blocks_allocated": 0,
            "blocks_in_use_final": 0,
            "naive_blocks_needed": 0,
        }

    total_blocks_allocated = 0

    def create_block(tokens=None):
        nonlocal total_blocks_allocated
        b = Block(total_blocks_allocated, tokens)
        total_blocks_allocated += 1
        return b

    # Step 0: Initialize beams with top beam_width tokens
    step0_log_probs = log_probs[0]
    top_indices = np.argsort(step0_log_probs)[::-1][:beam_width]

    current_beams = []
    for token_id in top_indices:
        token_id = int(token_id)  # Cast np.int64 to standard Python int
        score = float(step0_log_probs[token_id])
        blk = create_block([token_id])
        finished = token_id == eos_token
        current_beams.append(Beam(score, [blk], finished))

    # Perform beam search for subsequent steps
    for t in range(1, max_steps):
        if all(b.finished for b in current_beams):
            break

        candidates = []

        # Generate all expansion candidates
        for beam_idx, beam in enumerate(current_beams):
            if beam.finished:
                candidates.append({
                    "parent_idx": beam_idx,
                    "token": None,
                    "score": beam.score,
                    "finished": True,
                })
            else:
                step_log_probs = log_probs[t]
                for token_id in range(vocab_size):
                    cand_score = beam.score + float(step_log_probs[token_id])
                    cand_finished = token_id == eos_token
                    candidates.append({
                        "parent_idx": beam_idx,
                        "token": int(token_id),  # Ensure Python int
                        "score": cand_score,
                        "finished": cand_finished,
                    })

        # Sort candidates descending by score and pick top beam_width
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected_candidates = candidates[:beam_width]

        # Group selected candidates by parent index
        parent_to_children = {}
        for cand in selected_candidates:
            parent_to_children.setdefault(cand["parent_idx"], []).append(cand)

        new_beams = []

        # Process children for each parent
        for p_idx, parent_beam in enumerate(current_beams):
            children = parent_to_children.get(p_idx, [])
            num_children = len(children)

            if num_children == 0:
                for block in parent_beam.block_table:
                    block.ref_count -= 1
                continue

            for c_i, child in enumerate(children):
                is_last_child = c_i == num_children - 1

                if child["token"] is None:
                    if is_last_child:
                        child_blocks = parent_beam.block_table
                    else:
                        child_blocks = list(parent_beam.block_table)
                        for block in child_blocks:
                            block.ref_count += 1
                    new_beams.append(
                        Beam(child["score"], child_blocks, finished=True)
                    )
                else:
                    if is_last_child:
                        child_blocks = list(parent_beam.block_table)
                    else:
                        child_blocks = list(parent_beam.block_table)
                        for block in child_blocks:
                            block.ref_count += 1

                    last_block = child_blocks[-1]

                    if last_block.is_full(block_size):
                        new_blk = create_block([child["token"]])
                        child_blocks.append(new_blk)
                    else:
                        if last_block.ref_count > 1:
                            new_blk = create_block(
                                last_block.tokens + [child["token"]]
                            )
                            last_block.ref_count -= 1
                            child_blocks[-1] = new_blk
                        else:
                            last_block.tokens.append(child["token"])

                    new_beams.append(
                        Beam(
                            child["score"],
                            child_blocks,
                            finished=child["finished"],
                        )
                    )

        current_beams = new_beams

    # Sort final beams by score in descending order
    current_beams.sort(key=lambda x: x.score, reverse=True)

    final_sequences = [b.get_tokens() for b in current_beams]
    final_scores = [round(b.score, 4) for b in current_beams]

    final_blocks = set()
    for b in current_beams:
        for blk in b.block_table:
            final_blocks.add(blk.block_id)

    blocks_in_use_final = len(final_blocks)
    naive_blocks_needed = beam_width * int(math.ceil(max_steps / block_size))

    return {
        "sequences": final_sequences,
        "scores": final_scores,
        "total_blocks_allocated": total_blocks_allocated,
        "blocks_in_use_final": blocks_in_use_final,
        "naive_blocks_needed": naive_blocks_needed,
    }