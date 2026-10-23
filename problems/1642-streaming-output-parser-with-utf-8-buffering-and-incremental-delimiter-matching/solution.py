class StreamParser:
    def __init__(self, think_end="</think>", tool_open="<tool>", tool_close="</tool>"):
        self.think_end = think_end
        self.tool_open = tool_open
        self.tool_close = tool_close
        
        self.pi_think = self._compute_pi(think_end)
        self.pi_tool_open = self._compute_pi(tool_open)
        self.pi_tool_close = self._compute_pi(tool_close)
        
        self.phase = "reasoning"  # "reasoning", "text", "tool"
        self.j = 0
        self.held = ""
        self.leftover_bytes = b""
        self.tool_buffer = ""

    def _compute_pi(self, pattern):
        pi = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                pi[i] = length
                i += 1
            else:
                if length != 0:
                    length = pi[length - 1]
                else:
                    pi[i] = 0
                    i += 1
        return pi

    def feed(self, chunk):
        """Consume a bytes chunk; return the list of (kind, text) events it produced."""
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
            
        data = self.leftover_bytes + chunk
        try:
            data.decode('utf-8')
            valid_data, self.leftover_bytes = data, b''
        except UnicodeDecodeError as e:
            valid_data, self.leftover_bytes = data[:e.start], data[e.start:]
            
        text = valid_data.decode('utf-8', errors='replace')
        
        events = []
        current_content = []

        def flush_content(kind):
            if current_content:
                events.append((kind, "".join(current_content)))
                current_content.clear()

        for c in text:
            if self.phase == "reasoning":
                D = self.think_end
                pi = self.pi_think
            elif self.phase == "text":
                D = self.tool_open
                pi = self.pi_tool_open
            else:
                D = self.tool_close
                pi = self.pi_tool_close

            old_held = self.held
            temp_held = old_held + c
            
            j = self.j
            while j > 0 and c != D[j]:
                j = pi[j - 1]
            if c == D[j]:
                j += 1
            
            self.j = j
            self.held = temp_held[-j:] if j > 0 else ""
            
            if j == 0:
                confirmed = temp_held
            else:
                confirmed = temp_held[: len(old_held) + 1 - j]
                
            if confirmed:
                if self.phase == "tool":
                    self.tool_buffer += confirmed
                else:
                    current_content.append(confirmed)

            if j == len(D):
                self.j = 0
                self.held = ""
                
                if self.phase == "reasoning":
                    flush_content("reasoning")
                    self.phase = "text"
                elif self.phase == "text":
                    flush_content("text")
                    self.phase = "tool"
                elif self.phase == "tool":
                    if self.tool_buffer:
                        events.append(("tool", self.tool_buffer))
                        self.tool_buffer = ""
                    self.phase = "text"

        if self.phase != "tool":
            kind = "reasoning" if self.phase == "reasoning" else "text"
            flush_content(kind)
            
        return self._merge_events(events)

    def finish(self):
        """Flush held content; return the final events."""
        events = []
        if self.held:
            if self.phase == "tool":
                self.tool_buffer += self.held
            else:
                events.append(("reasoning" if self.phase == "reasoning" else "text", self.held))
            self.held = ""
            self.j = 0
            
        if self.phase == "tool" and self.tool_buffer:
            events.append(("tool", self.tool_buffer))
            self.tool_buffer = ""
            
        return self._merge_events(events)

    def _merge_events(self, events):
        if not events:
            return []
        merged = [events[0]]
        for kind, text in events[1:]:
            if merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + text)
            else:
                merged.append((kind, text))
        return merged