import math

def fcfs_preemption(requests, num_blocks, block_size):
    """Finish step and preemption count per request, and the peak number of swapped-out blocks."""
    num_reqs = len(requests)
    
    # States: 0: not arrived, 1: waiting, 2: running, 3: swapped, 4: done
    state = [0] * num_reqs
    cache_size = [0] * num_reqs
    output_produced = [0] * num_reqs
    finish_step = [0] * num_reqs
    preemption_counts = [0] * num_reqs
    
    peak_swapped_blocks = 0
    
    def get_blocks_needed(tokens):
        if tokens <= 0:
            return 0
        return math.ceil(tokens / block_size)

    def total_gpu_blocks_used():
        used = 0
        for i in range(num_reqs):
            if state[i] == 2:  # running
                used += get_blocks_needed(cache_size[i])
        return used

    def total_swapped_blocks():
        swapped_blocks = 0
        for i in range(num_reqs):
            if state[i] == 3:  # swapped out
                swapped_blocks += get_blocks_needed(cache_size[i])
        return swapped_blocks

    t = 0
    finished_count = 0
    
    while finished_count < num_reqs:
        newly_admitted = set()
        
        # 1. Arrive: requests with arrival_step <= t join waiting queue
        for i in range(num_reqs):
            if state[i] == 0 and requests[i][0] <= t:
                state[i] = 1  # waiting
                
        # 2. Swap in: go through swapped-out requests from earliest to latest arrival
        while True:
            swapped_candidates = [i for i in range(num_reqs) if state[i] == 3]
            if not swapped_candidates:
                break
            victim = min(swapped_candidates, key=lambda x: requests[x][0])
            needed = get_blocks_needed(cache_size[victim])
            free_blocks = num_blocks - total_gpu_blocks_used()
            if needed <= free_blocks:
                state[victim] = 2  # running
            else:
                break
                
        # 3. Admit: only if no request is swapped out, go through waiting queue
        swapped_candidates = [i for i in range(num_reqs) if state[i] == 3]
        if not swapped_candidates:
            waiting_candidates = [i for i in range(num_reqs) if state[i] == 1]
            for i in waiting_connections if False else waiting_candidates: # standard iteration
                prompt_len = requests[i][1]
                needed = get_blocks_needed(prompt_len)
                free_blocks = num_blocks - total_gpu_blocks_used()
                if needed <= free_blocks:
                    state[i] = 2  # running
                    cache_size[i] = prompt_len
                    output_produced[i] = 1  # prompt step produces 1 output token
                    newly_admitted.add(i)
                else:
                    break
                    
        # 4. Decode: every other running request, from earliest to latest arrival
        running_candidates = sorted([i for i in range(num_reqs) if state[i] == 2], key=lambda x: requests[x][0])
        
        for i in running_candidates:
            if state[i] != 2:
                continue
            
            # An admitted request does not also decode this step
            if i in newly_admitted:
                continue
                
            prompt_len, output_len = requests[i][1], requests[i][2]
            if output_produced[i] >= output_len:
                continue
                
            # Needs a new block if cache_size is a multiple of block_size
            if cache_size[i] > 0 and cache_size[i] % block_size == 0:
                needed = 1
                while num_blocks - total_gpu_blocks_used() < needed:
                    current_running = sorted([r for r in range(num_reqs) if state[r] == 2], key=lambda x: requests[x][0])
                    if not current_running:
                        break
                    victim = current_running[-1]  # latest arrival
                    state[victim] = 3  # swapped out
                    preemption_counts[victim] += 1
                    
                    if victim == i:
                        break
                
                if state[i] != 2:
                    continue
            
            cache_size[i] += 1
            output_produced[i] += 1
            
        peak_swapped_blocks = max(peak_swapped_blocks, total_swapped_blocks())
        
        # 5. Finish: every running request that has produced output_len tokens frees its blocks and records t
        for i in range(num_reqs):
            if state[i] == 2:
                if output_produced[i] >= requests[i][2]:
                    state[i] = 4  # done
                    finish_step[i] = t
                    finished_count += 1
                    
        t += 1

    return {
        'finish': finish_step,
        'preemptions': preemption_counts,
        'peak_swapped_blocks': peak_swapped_blocks
    }